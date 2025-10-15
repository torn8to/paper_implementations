from local_map_management import OdometryWrapper
from kitti360_dataloader import Kitti360LidarData
from density import generate_density_image_map
import matplotlib.pyplot as plt
from loop_closure_detector import LoopClosureDetector
import numpy as np
import timeit
import pickle
from tqdm import tqdm


def main():
  max_frames = 9000
  movement_threshold = 25.0
  kiss_pipeline = OdometryWrapper()
  data_loader = Kitti360LidarData()
  map_closure_detector = LoopClosureDetector()

  last_density_map_pose = kiss_pipeline.get_current_position()
  for _ in tqdm(range(max_frames)):
    if not data_loader.has_next():
      return
    cloud = data_loader.retrieve_next_frame()
    kiss_pipeline.register_frame(cloud[:, :3])
    # print(f"{t} time to executeframe win map size {kiss_pipeline.get_local_map_cloud().shape[0]}")
    if (np.linalg.norm(last_density_map_pose[:3, 3] - kiss_pipeline.get_current_position()[:3, 3]) > 25.0).any():
      local_map_cloud = kiss_pipeline.get_local_map_cloud()
      last_density_map_pose = kiss_pipeline.get_current_position()
      map_closure_detector.add_new_loop_closure_entry_from_cloud(local_map_cloud, position=last_density_map_pose)
    if _ % 100 == 99:
      print(len(map_closure_detector.list_of_loop_closure_entrys))

  map_cloud = kiss_pipeline.get_local_map_cloud()
  pos = kiss_pipeline.get_current_position()[0:3, 3]
  list_of_images = [(i.img, i.position, i.cloud) for i in map_closure_detector.list_of_loop_closure_entrys]
  pickle.dump(list_of_images, open("loop_closure_data_xl.pkl", "wb"))

  # Generate final density map for visualization
  dens_img = generate_density_image_map(map_cloud, position=pos, resolution=0.5)
  plt.imshow(dens_img.img)
  plt.colorbar(label="point_density")
  plt.show()


if __name__ == "__main__":
  main()

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
    if (np.linalg.norm(last_density_map_pose[:3, 3] - kiss_pipeline.get_current_position()[:3, 3]) > 5.0).any():
      local_map_cloud = kiss_pipeline.get_local_map_cloud()
      last_density_map_pose = kiss_pipeline.get_current_position()
      map_closure_detector.match_and_add_new(local_map_cloud, position=last_density_map_pose)


if __name__ == "__main__":
  main()

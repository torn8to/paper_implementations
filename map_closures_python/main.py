from local_map_management import OdometryWrapper
from kitti360_dataloader import Kitti360LidarData
from density import generate_density_image_map
from pipeline import OptimizationPipeline, OptimizationPipelineConfig
import matplotlib.pyplot as plt
from loop_closure_detector import LoopClosureDetector
import numpy as np
import timeit
import pickle
from tqdm import tqdm


opc = OptimizationPipelineConfig(
    max_points_per_voxel=20,
    alpha=1.0,
    max_hamming_distance=35
)


def main():
    max_frames = 9000
    movement_threshold = 25.0
    kiss_pipeline = OdometryWrapper()
    data_loader = Kitti360LidarData()
    pipeline = OptimizationPipeline(opc)

    last_density_map_pose = kiss_pipeline.get_current_position()

    for _ in tqdm(range(max_frames)):
        if not data_loader.has_next():
            return

        cloud = data_loader.retrieve_next_frame()
        cloud_xyz = cloud[:,:3]
        kiss_pipeline.register_frame(cloud_xyz)
        odom_position = kiss_pipeline.get_current_position()
        pipeline.update(cloud_xyz, odom_position)


if __name__ == "__main__":
    main()

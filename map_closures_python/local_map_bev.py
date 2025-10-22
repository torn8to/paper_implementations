from kiss_icp.mapping import VoxelHashMap
from kiss_icp.voxelization import Voxelize
from density import generate_density_image_map
import numpy as np


class LocalMapBev:
    def __init__(self, max_points_per_voxel: int = 13,
                 max_range: float = 100.0,
                 voxel_size: float = 1.0,
                 alpha: float = 0.5):

        self.max_range: float = max_range
        self.voxel_size: float = voxel_size
        self.alpha: float = alpha
        self.map = VoxelHashMap(
            voxel_size=self.voxel_size,
            max_distance=self.max_distance,
            max_points_per_voxel=max_points_per_voxel)
        self.current_position = np.eye(4)

    def clear(self):
        self.map.empty()

    def empty(self):
        self.map.clear()

    def add_reading(self, cloud: np.ndarray, position: np.ndarray,):
        sampled = Voxelize(cloud, self.voxel_size * self.alpha)
        self.map.update(sampled, position)
        self.current_position = position

    def cloud(self):
        return self.map.point_cloud()

    def bev_density_image(self, resolution):
        return generate_density_image_map(self.cloud(), resolution,
                                          self.max_range,
                                          self.current_position[:3, 3])

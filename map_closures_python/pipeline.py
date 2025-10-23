from local_map_bev import LocalMapBev
from loop_closure_detector import LoopClosureDetector
from pgo import PGO
from dataclasses import dataclass
from itertools import count
from collections import namedtuple
from typing import Optional
import numpy as np

LoopClosureIds = namedtuple("LoopClosureIds", ["id1", "id2", "estimate", "covariance"])


@dataclass
class OptimizationPipelineConfig:
    max_points_per_voxel: int = 27
    max_range: float = 100.0
    voxel_size: float = 0.5
    alpha: float = 0.5
    bev_resolution: float = 0.5
    max_hamming_distance: int = 35
    n_features = 500
    pose_diff: float = 5.0
    matching_point_threshold_alpha: int = 25
    matching_point_threshold_beta: int = 25


class OptimizationPipeline:
    _counter = count(0)
    def __init__(self, config: OptimizationPipelineConfig = OptimizationPipelineConfig()):
        self.config = config
        self.bev_resolution = config.bev_resolution
        self.graph_optimizer: PGO = PGO()
        self.last_registered_pose = np.eye(4)
        self.lcd: LoopClosureDetector = LoopClosureDetector(max_hamming_distance=config.max_hamming_distance,
                                                            n_features=config.n_features)
        self.lmb: LocalMapBev = LocalMapBev(max_points_per_voxel=config.max_points_per_voxel,
                                            max_range=config.max_range,
                                            voxel_size=config.voxel_size,
                                            alpha=config.alpha)

        self.loop_closure_tags: list[LoopClosureIds] = []

    def update(self, cloud, new_odom, covariance_matrix: Optional[np.ndarray]=None) -> Optional[LoopClosureIds]:
        id = next(self._counter)
        print(id)
        self.graph_optimizer.add_odom(new_odom, self.lmb.position(), covariance_matrix)
        self.lmb.add_reading(cloud, new_odom)
        if np.linalg.norm(new_odom[:3,3] - self.last_registered_pose[:3,3]) > self.config.pose_diff:
            bev_density_img = self.lmb.bev_density_image(self.bev_resolution)
            lcdids = self.lcd.match_and_add_new(bev_density_img, self.lmb.position(), id)
            self.last_registered_pose = self.lmb.position()
            return lcdids
        return None

    def get_all_loop_closure_ids(self):
        return self.loop_closure_tags

    def point_cloud(self):
        self.lmb.cloud()

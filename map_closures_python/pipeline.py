from local_map_bev import LocalMapBev
from loop_closure_detector import LoopClosureDetector
from pgo import PGO
from dataclasses import dataclass
from itertools import count
from collections import namedtuple
import numpy as np

LoopClosureIds = namedtuple("LoopClosureIds", ["id1", "id2", "estimate", "covariance"])


class OptimizationPipelineConfig:
    max_range: float
    voxel_size: float
    alpha: float = 0.5
    max_hamming_distance:int = 35


class OptimizationPipeline:
    def __init__(self, config: OptimizationPipelineConfig = OptimizationPipelineConfig):
        self.bev_resolution
        self.graph_optimizer: PGO = PGO()
        self.lcd: LoopClosureDetector = LoopClosureDetector()
        self.lmb: LocalMapBev = LocalMapBev()
        self.loop_closure_tags: list[LoopClosureIds] = []

    def update(self, cloud, new_odom, covariance_matrix: Optional[np.ndarray]=None) -> Optional[LoopClosureIds]:
        self.graph_optimizer.add_odom(new_odom, self.lmb.position(), covariance_matrix)
        self.lmb.add_reading(cloud, new_odom)
        bev_density_img = self.lmb.bev_density_image(self.bev_resolution)
        lcdids = self.lcd.match_and_add_new(bev_density_img, self.lmb.position())
        return lcdids

    def get_all_loop_closure_ids(self):
        return self.loop_closure_tags

    def point_cloud(self):
        self.lmb.cloud()

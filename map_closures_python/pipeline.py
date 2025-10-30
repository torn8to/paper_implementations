from local_map_bev import LocalMapBev
from loop_closure_detector import LoopClosureDetector
from pgo import PGO
from dataclasses import dataclass
from itertools import count
from collections import namedtuple
from typing import Optional, List, Iterator
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
    pose_diff: float = 15.0
    matched_points_thredhold_alpha: int = 25
    ransac_matching_threshold: int = 25

class OptimizationPipeline:
    """Maintain a local voxel map and render BEV density images.

    Parameters
    ----------
    max_points_per_voxel : int, default: 13
        Maximum number of points to retain per voxel in the map.
    max_range : float, default: 100.0
        Maximum distance (meters) from the current pose to keep points in the map.
    voxel_size : float, default: 1.0
        Edge length (meters) of voxels used by the underlying map.
    alpha : float, default: 0.5
        Factor applied to ``voxel_size`` for pre-insertion downsampling.

    Attributes
    ----------
    _counter: Iterator
        for keeping track of odometry measurements generating a vertex id with each new measurement
    _map_counter: Iterator
        this id is for keeping track of loop_closure_ids in the matcher to odom_id
    config: OptimizationPipelinConfig
        the config passed to the object
    graph_optimizer: PGO
        the object encompasses the pose graph optimizer
    observation_to_odom_map: dict
        a dictionary to map loop_closure entries to odom readings
    lcd : LoopClosureDetector
        this object encompasses a way to get optimal map closures stores the  
        keypoint matching and ransac portion of the pipeline
    lmb: LocalMapBev
        manages the sparse voxel hashmap to perform point comparisons and generate density birds eye view images
    current_position : numpy.ndarray of shape (4, 4)
        Latest homogeneous transform of the sensor in the world frame.
    self.loop_closure_tags: list[LoopClosureIds]
        a list of tuple storing ids of vertexes of a loop closure for storage mostly for visualization
    """
    _counter: Iterator = count(0)

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
        """Update the state of the loop closure
        cloud: np.ndarray
            a point cloud
        new_odom: np.ndarray
            a new odometry reading to update the backend
        covariance_matrix: Optional[np.ndarray]
            if the covariance matrix is not None it gets added to the pose graph optimizer setting the wieight 
            recomended for better pose graph optimization
        """
        id = next(self._counter)

        self.lmb.add_reading(cloud, new_odom)
        self.graph_optimizer.add_odom(id, new_odom, self.lmb.position(), covariance_matrix)
        new_loop_closures_added: List[LoopClosureIds] = []
        if np.linalg.norm(new_odom[:3, 3] - self.last_registered_pose[:3, 3]) > self.config.pose_diff:
            bev_density_img = self.lmb.bev_density_image(self.bev_resolution)
            lcdids = self.lcd.match_and_add_new(bev_density_img, self.lmb.position(), id)
            self.last_registered_pose = self.lmb.position()
            if lcdids is not None:
                for i in lcdids:
                    new_loop_closures_added.append((id, i[0], np.linalg.norm(i[1].t)))
                    self.graph_optimizer.add_loop_closure_edge(id, i[0], i[2][0])
            if len(new_loop_closures_added) > 0:
                self.graph_optimizer.optimize(iterations=20)
        return new_loop_closures_added

    def pgo_position(self):
        """returns the slam position of the last added vertex
        """
        return self.graph_optimizer.position()

    def odom_position(self):
        """returns the last odometry position given as an update 
        """
        return self.last_registered_pose

    def get_vertices_np(self) -> List[np.ndarray]:
        """returns the vertices by vertex id as a list of numpy arrays
        """
        return self.graph_optimizer.get_vertex_poses()

    def get_all_loop_closure_ids(self) -> List[LoopClosureIds]:
        """ returns all loop loop closures
        """
        return self.loop_closure_tags

    def point_cloud(self) -> np.ndarray:
        self.lmb.cloud()

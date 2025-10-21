import g2o
import numpy as np
from itertools import count
from typing import Optional


'''
for i in dir(g2o.VertexSE3):
    print(i)
'''
class PGO:
    def __init__(self):
        self._optimizer = g2o.SparseOptimizer()
        self._solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
        self._algorithm = g2o.OptimizationAlgorithmLevenberg(self._solver)
        self._optimizer.set_algorithm(self._algorithm)
        self.vertex_count = count(0)
        vertex_se3 = g2o.VertexSE3()
        vertex_se3.set_estimate(g2o.Isometry3d())
        vertex_se3.set_id(next(self.vertex_count))
        vertex_se3.set_fixed(True)
        self._optimizer.add_vertex(vertex_se3)

    def add_odom(self, current_pose, last_pose, position_covariance_matrix:np.ndarray= np.eye(6))->None:
        relative_pose = current_pose @ np.linalg.inv(last_pose)
        estimate_isometry = g2o.Isometry3d(relative_pose[:3, :3], relative_pose[3,:3])
        current_isometry = g2o.Isometry3d(current_pose[:3, :3], current_pose[3, :3])
        vertex_se3 = g2o.VertexSE3()
        current_vertex_id = next(self.vertex_count)
        vertex_se3.set_id(current_vertex_id)
        vertex_se3.set_estimate(current_isometry)

        edge_se3 = g2o.EdgeSE3()
        edge_se3.set_vertex(0, self.get_vertex_by_id(current_vertex_id-1))
        edge_se3.set_vertex(1, self.get_vertex_by_id(current_vertex_id))
        edge_se3.set_estimate(estimate_isometry)
        information_matrix = np.linalg.inverse(position_covariance_matrix)
        edge_se3.set_information(information_matrix)
        self._optimizer.add_vertex(vertex_se3)
        self._optimizer.add_edge(edge_se3)

    def add_loop_closure_edge(self, id_: int, match_id: int, relative_pose: np.ndarray, covariance_matrix: Optional[np.ndarray] =None):
        relative_pose_g2o = g2o.Isometry3d(relative_pose[:3, :3],
                                           relative_pose[3, :3])
        edge_se3 =  g2o.EdgeSE3()
        edge_se3.set_vertex(0, id)
        edge_se3.set_vertex(1, match_id)
        edge_se3.set_measurement(relative_pose_g2o)
        if covariance_matrix != None:
            edge_se3.set_information(np.linalg.inv(covariance_matrix))
        self._optimizer.add_edge(edge_se3)

    def get_vertex_by_id(self, id: int) -> g2o.VertexSE3:
        return self._optimizer.vertex(id)

    def optimize(self, iterations: int = 500, verbose: bool = True):
        self._optimizer.initialize_optimization()
        self._optimizer.set_verbose(verbose)
        self._optimizer.optimize(500)

    def save(self, name:str="pose_graph.g2o"):
        self._optimizer.save(name)

import g2o
import numpy as np
from typing import Optional


class SimplePGO:
  def __init__(self):
    self._optimizer = g2o.SparseOptimizer()
    self._solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
    self._algorithm = g2o.OptimizationAlgorithmLevenberg(self._solver)
    self._optimizer.set_algorithm(self._algorithm)
    self.vertex_count = 0

  def get_vertex_by_id(self, id) -> Optional[g2o.EdgeSE3]:
    return optimizer

  def add(self, new_relative_pose: np.ndarray, position_covariance_matrix: np.ndarray):
    vertex_se3 = g2o.VertexSE3()
    last_vertex = self.get_vertex_by_id()
    edge_ = g2o.EdgeSE3
    information_matrix = np.linalg.inverse(position_covariance_matrix)
    self.vertex_count = self.vertex_count + 1

  def add_loop_closure_edge(self, vertex_id, matched_vertex_id, relative_pose):
    loop_closure_edge = g2o.EdgeSE3()
    loop_closure_edge.set(0, self.get_vertex_by_id(vertex_id))
    loop_closure_edge.set(1, self.get_vertex_by_id(matched_vertex_id))
    loop_closure_edge.set_measurement(g2o.Isometry3d(relative_pose[:3, :3], relative_pose[:3, 3]))
    self._optimizer.add_edge(loop_closure_edge)

  def get_list_of_poses(self) -> list[np.ndarray]:
    list_of_poses: List[np.ndarray] = []
    for v_id, vertex in self._optimizer.vertices().items():
      P = np.eye(4)
      p[:3, :3] = vertex.estimate().rotation()
      p[:3, 3] = vertex.estimate().translation()
      list_of_poses.append(P)
    return list_of_poses

  def optimize(self):
    self._optimizer.optimize(50)

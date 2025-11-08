"""Pose Graph Optimization (PGO) utilities using g2o.

This module provides a thin wrapper around g2o for building and optimizing
SE(3) pose graphs with odometry and loop-closure constraints.
"""

import g2o
import numpy as np
from itertools import count
from typing import Optional


class PGO:
    """Minimal pose-graph optimizer.

    Attributes
    ----------
    _optimizer : g2o.SparseOptimizer
        Underlying g2o optimizer instance.
    _solver : g2o.BlockSolverX
        Block solver used by the optimizer.
    _algorithm : g2o.OptimizationAlgorithmLevenberg
        Optimization algorithm.
    """

    def __init__(self):
        self.latest_vertex_id = 0
        self._optimizer = g2o.SparseOptimizer()
        self._solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
        self._algorithm = g2o.OptimizationAlgorithmLevenberg(self._solver)
        self._optimizer.set_algorithm(self._algorithm)
        vertex_se3 = g2o.VertexSE3()
        vertex_se3.set_estimate(g2o.Isometry3d())
        vertex_se3.set_id(0)
        vertex_se3.set_fixed(True)
        self._optimizer.add_vertex(vertex_se3)

    def add_odom(self, id: int, current_odometry_pose, last_odometry_pose, covariance_matrix: Optional[np.ndarray] = None) -> None:
        """Add an odometry edge between consecutive poses and a new vertex.

        Parameters
        ----------
        current_pose : numpy.ndarray of shape (4, 4)
            Current pose T_wc in homogeneous coordinates.
        last_pose : numpy.ndarray of shape (4, 4)
            Previous pose T_wl in homogeneous coordinates.
        covariance_matrix : numpy.ndarray of shape (6, 6), optional
            Measurement covariance for the SE3 constraint.
        Returns
        -------
        None
        """
        relative_pose = np.linalg.inv(last_odometry_pose) @ current_odometry_pose

        estimate_isometry = g2o.Isometry3d(relative_pose[:3, :3], relative_pose[:3, 3])
        current_isometry = g2o.Isometry3d(current_odometry_pose[:3, :3], current_odometry_pose[:3, 3])

        vertex_se3 = g2o.VertexSE3()
        vertex_se3.set_id(id)
        vertex_se3.set_estimate(current_isometry)

        edge_se3 = g2o.EdgeSE3()
        edge_se3.set_vertex(0, self.get_vertex_by_id(id - 1))
        edge_se3.set_vertex(1, self.get_vertex_by_id(id))
        edge_se3.set_measurement(estimate_isometry)
        if covariance_matrix is not None:
            edge_se3.set_information(np.linalg.inv(covariance_matrix))

        self._optimizer.add_vertex(vertex_se3)
        self._optimizer.add_edge(edge_se3)
        self.latest_vertex_id = id

    def add_loop_closure_edge(self, id: int, match_id: int, relative_pose: np.ndarray, covariance_matrix: Optional[np.ndarray] = None):
        """Add a loop-closure constraint between two existing vertices.

        Parameters
        ----------
        id_ : int
            Source vertex id.
        match_id : int
            Target vertex id to connect with a loop-closure edge.
        relative_pose : numpy.ndarray of shape (4, 4)
            Relative transform T_im from source to target.
        covariance_matrix : numpy.ndarray of shape (6, 6), optional
            Measurement covariance for the SE3 constraint.

        Returns
        -------
        None
        """
        relative_pose_g2o = g2o.Isometry3d(relative_pose[:3, :3], relative_pose[:3, 3])

        edge_se3 = g2o.EdgeSE3()
        edge_se3.set_vertex(0, self.get_vertex_by_id(id))
        edge_se3.set_vertex(1, self.get_vertex_by_id(match_id))
        edge_se3.set_measurement(relative_pose_g2o)

        if covariance_matrix != None:
            edge_se3.set_information(np.linalg.inv(covariance_matrix))
        self._optimizer.add_edge(edge_se3)

    def optimize(self, iterations: int = 500, verbose: bool = False):
        """Run pose-graph optimization.

        Parameters
        ----------
        iterations : int, default: 500
            Maximum iterations to run.
        verbose : bool, default: True
            Whether to print g2o solver logs.

        Returns
        -------
        None
        """
        self._optimizer.initialize_optimization()
        self._optimizer.set_verbose(verbose)
        self._optimizer.optimize(iterations)

    def save(self, name: str = "pose_graph.g2o"):
        """Serialize the current graph to a .g2o file.

        Parameters
        ----------
        name : str, default: "pose_graph.g2o"
            Output file path.

        Returns
        -------
        None
        """
        self._optimizer.save(name)

    def get_vertex_by_id(self, id: int) -> g2o.VertexSE3:
        """Get a vertex by id.

        Parameters
        ----------
        id : int
            Vertex identifier.

        Returns
        -------
        g2o.VertexSE3
        """
        return self._optimizer.vertex(id)

    def get_vertex_poses(self) -> list[np.ndarray]:
        """Return all vertex poses as 4x4 arrays.

        Returns
        -------
        list of numpy.ndarray
            List of SE3 homogeneous matrices, in vertex id order.
        """
        return [self._optimizer.vertex(i).estimate().matrix() for i in sorted(self._optimizer.vertices().keys())]

    def position(self):
        """Return all vertex poses as 4x4 arrays.

        Returns
        -------
        numpy.ndarray
            SE3 homogeneous transformation matrix of the latest vertex
        """
        return self._optimizer.vertex(self.latest_vertex_id).estimate().matrix()

    def save_optimizer(self):
        self._optimizer

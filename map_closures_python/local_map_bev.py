"""Local BEV map utilities.

This module provides a thin wrapper around a kiss-ICP VoxelHashMap to
accumulate lidar point clouds in a local frame and to render a bird's-eye
view (BEV) density image around the current pose.
"""

from kiss_icp.mapping import VoxelHashMap
from kiss_icp.voxelization import voxel_down_sample
from density import generate_density_image_map
import numpy as np


class LocalMapBev:
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
    map : kiss_icp.mapping.VoxelHashMap
        Underlying spatial hash voxel map.
    current_position : numpy.ndarray of shape (4, 4)
        Latest homogeneous transform of the sensor in the world frame.
    max_range : float
    voxel_size : float
    alpha : float
        a coefficient that adjusts the size of maps compared to

    Examples
    --------
    >>> import numpy as np
    >>> lm = LocalMapBev(voxel_size=1.0, max_range=50.0)
    >>> cloud = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
    >>> T = np.eye(4, dtype=np.float32)
    >>> lm.add_reading(cloud, T)
    >>> isinstance(lm.cloud(), np.ndarray)
    True
    """

    def __init__(self, max_points_per_voxel: int = 13, max_range: float = 100.0, voxel_size: float = 1.0, alpha: float = 0.5):
        self.max_range: float = max_range
        self.voxel_size: float = voxel_size
        self.alpha: float = alpha
        self.map = VoxelHashMap(voxel_size=self.voxel_size, max_distance=self.max_range, max_points_per_voxel=max_points_per_voxel)
        self.current_position = np.eye(4)

    def clear(self):
        """Remove all points from the internal voxel map.

        Returns
        -------
        None
        """
        self.map.empty()

    def empty(self):
        """Alias of ``clear``; removes all points from the map.

        Returns
        -------
        None
        """
        self.map.clear()

    def add_reading(
        self,
        cloud: np.ndarray,
        position: np.ndarray,
    ):
        """Insert a point cloud measurement into the local map.

        Parameters
        ----------
        cloud : numpy.ndarray of shape (N, 3)
            Point cloud (meters). Coordinates must be consistent with ``position``.
        position : numpy.ndarray of shape (4, 4)
            Homogeneous transform of the sensor pose in the world frame.

        Returns
        -------
        None

        Notes
        -----
        The input cloud is voxel down-sampled with voxel size ``voxel_size * alpha``
        prior to insertion.
        """
        sampled = voxel_down_sample(cloud, self.voxel_size * self.alpha)
        self.map.update(sampled, position)
        self.current_position = position

    def cloud(self):
        """Return the current map as a point cloud.

        Returns
        -------
        numpy.ndarray of shape (M, 3)
            Concatenated points stored in the voxel map.
        """
        return self.map.point_cloud()

    def bev_density_image(self, resolution) -> np.ndarray:
        """Compute a bird's-eye-view density image from the local map.

        Parameters
        ----------
        resolution : float
            Grid resolution in meters per pixel used for the BEV image.

        Returns
        -------
        numpy.ndarray
            2D BEV density image centered at the current position.
        """
        return generate_density_image_map(self.cloud(), resolution, self.max_range, position=self.current_position[:3, 3])

    def position(self) -> np.ndarray:
        """Return the latest sensor pose.

        Returns
        -------
        numpy.ndarray of shape (4, 4)
            Homogeneous transform of the current sensor pose in the world frame.
        """
        return self.current_position

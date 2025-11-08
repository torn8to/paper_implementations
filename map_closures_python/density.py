"""Density map utilities for BEV representations.

This module provides helpers to convert 3D point clouds into 2D
bird's-eye-view (BEV) density images, along with small utilities for
voxelization and 2D affine warping.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import affine_transform


class DensityMap:
  """Container for a BEV density image.

  Parameters
  ----------
  img : numpy.ndarray
      Grayscale 2D image, typically uint8, where larger values indicate
      higher point density in the corresponding grid cell.

  Attributes
  ----------
  img : numpy.ndarray
      The underlying image array.
  """
  def __init__(self, img: np.ndarray):
    self.img = img




def voxelizeXY(mat: np.ndarray, resolution: float) -> np.ndarray:
  """Quantize XY coordinates into voxel indices.

  Parameters
  ----------
  mat : numpy.ndarray of shape (N, >=2)
      Input array containing at least XY columns.
  resolution : float
      Size of each grid cell in meters per pixel.

  Returns
  -------
  numpy.ndarray of shape (N, 2), dtype int32
      Integer grid coordinates for each point.

  Examples
  --------
  >>> import numpy as np
  >>> pts = np.array([[0.2, 0.2, 0.0], [1.0, 1.0, 0.0]])
  >>> voxelizeXY(pts, 0.5)
  array([[0, 0],
         [2, 2]], dtype=int32)
  """
  return (mat[:, :2] / resolution).astype(np.int32)

def first_stage_debug(map_cloud, position, max_range, resolution):
  map_cloud_centered = map_cloud - position
  cols, rows = int(max_range * 2 / resolution), int(max_range * 2 / resolution)  # add one for padding
  map_points2d = voxelizeXY(map_cloud_centered, resolution)
  map_points2d[:, 0] = map_points2d[:, 0] - rows // 2
  map_points2d[:, 1] = cols // 2 - map_points2d[:, 1]
  map_counts = np.unique(map_points2d, return_counts=True, axis=0)
  count_image_raw = np.zeros((rows + 2, cols + 2), dtype=np.int32)  # create image with padding
  return count_image_raw, map_counts

def second_stage_debug(count_image_raw, lower_clipping_threshold):
  count_image_float = np.array(count_image_raw, dtype=np.float32)
  max = count_image_float.max()
  min = count_image_float.min()
  img_norm_float = (count_image_float - min) / (max - min)
  img_norm_float[img_norm_float < lower_clipping_threshold] = 0.00
  img_norm_float = img_norm_float * 127
  img_formatted = img_norm_float.astype(np.uint8)
  return img_formatted

def generate_density_image_map(
  map_cloud: np.ndarray, resolution: float, max_range=100, lower_clipping_threshold: float = 0.05, position=np.zeros(3)
) -> DensityMap:
  """Generate a BEV density image from a point cloud.

  Parameters
  ----------
  map_cloud : numpy.ndarray of shape (N, 3)
      Input point cloud in meters.
  resolution : float
      Grid resolution in meters per pixel.
  max_range : float, default: 100
      Half-size of the square BEV window in meters. Output image is approximately
      ``(2*max_range/resolution)`` in both dimensions (with small padding).
  lower_clipping_threshold : float, default: 0.05
      Values below this normalized density are set to 0.
  position : numpy.ndarray of shape (3,), default: np.zeros(3)
      Center position (x, y, z). The BEV will be centered at this XY.

  Returns
  -------
  DensityMap
      Container holding the uint8 BEV image.

  Notes
  -----
  - Densities are computed by counting points per grid cell, then min-max
    normalized and scaled to [0, 127].
  - Coordinates are shifted to place ``position`` at the image center with a
    conventional image axis layout.

  Examples
  --------
  >>> import numpy as np
  >>> cloud = np.array([[0.,0.,0.], [0.1, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
  >>> dm = generate_density_image_map(cloud, resolution=0.5, max_range=2.0)
  >>> isinstance(dm.img, np.ndarray)
  True
  """
  assert position.shape[0] == 3 and len(position.shape) == 1, "position.shape is not right size"

  count_image_raw, map_counts = first_stage_debug(map_cloud, position, max_range, resolution)


  for i in range(map_counts[0].shape[0]):
    pixel = map_counts[0][i]
    pixel_count = map_counts[1][i]
    count_image_raw[pixel[0], pixel[1]] = pixel_count

  img_formatted = second_stage_debug(count_image_raw, lower_clipping_threshold)

  return DensityMap(img_formatted)


def affine_transform_2d(img: np.ndarray, rotation_matrix: np.ndarray, T):
  """Apply a 2D affine transform to an image.

  Parameters
  ----------
  img : numpy.ndarray
      Input 2D image.
  rotation_matrix : numpy.ndarray of shape (2, 2)
      Rotation (or linear) component of the transform.
  T : numpy.ndarray of shape (2,)
      Translation vector.

  Returns
  -------
  numpy.ndarray
      Warped output image.
  """
  r_inv = rotation_matrix.T
  offset = -r_inv * T
  return affine_transform(input=img, matrix=r_inv, offset=offset, mode="constant", cval=0.0)

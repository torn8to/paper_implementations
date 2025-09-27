import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import affine_transform


class DensityMap:
  def __init__(self, img: np.ndarray):
    self.img = img


"""
get voxels for all points in a cloud
returns a numpy array of the same shape

"""


def voxelizeXY(mat: np.ndarray, resolution: float) -> np.ndarray:
  return (mat[:, :2] / resolution).astype(np.int32)


def generate_density_image_map(
  map_cloud: np.ndarray, resolution: float, max_range=100, lower_clipping_threshold: float = 0.05, position=np.zeros(3)
) -> DensityMap:
  assert position.shape[0] == 3 and len(position.shape) == 1, "position.shape is not right size"
  map_cloud_centered = map_cloud - position
  cols, rows = int(max_range * 2 / resolution), int(max_range * 2 / resolution)  # add one for padding
  map_points2d = voxelizeXY(map_cloud_centered, resolution)
  map_points2d[:, 0] = map_points2d[:, 0] - rows // 2
  map_points2d[:, 1] = cols // 2 - map_points2d[:, 1]
  map_counts = np.unique(map_points2d, return_counts=True, axis=0)
  count_image_raw = np.zeros((rows + 2, cols + 2), dtype=np.int32)  # create image with padding

  for i in range(map_counts[0].shape[0]):
    pixel = map_counts[0][i]
    pixel_count = map_counts[1][i]
    count_image_raw[pixel[0], pixel[1]] = pixel_count

  count_image_float = np.array(count_image_raw, dtype=np.float32)
  max = count_image_float.max()
  min = count_image_float.min()
  img_norm_float = (count_image_float - min) / (max - min)
  img_norm_float[img_norm_float < lower_clipping_threshold] = 0.00
  img_norm_float = img_norm_float * 127
  img_formatted = img_norm_float.astype(np.uint8)
  return DensityMap(img_formatted)


def affine_transform_2d(img: np.ndarray, rotation_matrix: np.ndarray, T):
  r_inv = rotation_matrix.T
  offset = -r_inv * T
  return affine_transform(input=img, matrix=r_inv, offset=offset, mode="constant", cval=0.0)

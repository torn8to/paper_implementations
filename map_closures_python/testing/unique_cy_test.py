from cython_modules.unique_cy import unique_voxel_map
import numpy as np
from numpy.testing import assert_array_equal
import random
import unittest


def generate_random_voxel_points(number_of_voxels: int = 1000, bounds: tuple[int, int] = (0, 399)) -> np.ndarray:
    """A Testing Helper function to build"""
    return np.random.randint(low=bounds[0], high=bounds[1], size=(number_of_voxels, 2), dtype=np.int32)


def unique_encoding(map_points2d, max_range, resolution):
    """this is the default implementation were testing cython modules agains against"""
    cols, rows = int((max_range * 2) / resolution), int((max_range * 2) / resolution)
    map_counts = np.unique(map_points2d, return_counts=True, axis=0)
    count_image_raw = np.zeros((rows, cols), dtype=np.int32)  # create image with padding
    return count_image_raw, map_counts


def generate_bev_image(xy_voxel_points: np.ndarray, resolution: float = 0.5, max_range: float = 100.0, method: str = "numpy") -> np.ndarray:
    assert method in ["cython", "numpy"], " bad method passed to this"
    if method == "numpy":
        count_image_raw, map_counts = unique_encoding(xy_voxel_points, max_range, resolution)
        for i in range(map_counts[0].shape[0]):
            pixel = map_counts[0][i]
            pixel_count = map_counts[1][i]
            count_image_raw[pixel[0], pixel[1]] = pixel_count
        return count_image_raw
    else:
        count_image_raw = unique_voxel_map(xy_voxel_points, max_range, resolution)
        return count_image_raw


class testingUniqueOperation(unittest.TestCase):
    """that replacing numpy unique with a cython function provides the same results over images"""

    def test_fuzz_unique_operation_equivalent_numpy(self):
        """checking equivalent output between methods"""

        np.random.seed(42)
        random.seed(42)

        num_points = [random.randint(10000, 100000) for i in range(10)]

        # set size to 400 by 400
        max_range = 100.0
        resolution = 0.5
        for n in num_points:
            voxel_points = generate_random_voxel_points(n)
            numpy_output = generate_bev_image(voxel_points, resolution, max_range, method="numpy")
            cython_output = generate_bev_image(voxel_points, resolution, max_range, method="cython")
            assert_array_equal(cython_output.flatten(), numpy_output.flatten())


if __name__ == "__main__":
    unittest.main()

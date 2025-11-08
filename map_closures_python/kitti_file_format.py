import csv
import numpy as np


def save_poses_as_kitti(poses: list[np.ndarray], filename: str = "./kitti_file.txt"):
    """saves Homogenous transformation_matrix to 3x4 to the kitti file format

    Parameters
    ----------
    max_points_per_voxel : list[np.ndarray]
        The poses to be saved in the file
    filename : str, default: ./kitti_file.txt
        the filename/location the file is saved at
    """
    with open(filename, "w") as kitti_file:
        kitti_writer = csv.writer(kitti_file, delimiter=" ")
        for pose in poses:
            kitti_writer.writerow(pose[:3].reshape(-1).tolist())

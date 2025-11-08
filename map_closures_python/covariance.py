import numpy as np
from scipy.spatial.transform import Rotation
# import open3d as o3d


def odom_covariance_calculation(relative_transform: np.ndarray, factor=0.02):
    rot = Rotation.from_matrix(relative_transform[:3, :3])
    euler_xyz = rot.as_rotvec(degrees=False)
    # print(euler_xyz)
    return factor * np.diag([euler_xyz[0], euler_xyz[1], euler_xyz[2], relative_transform[0, 3], relative_transform[1, 3], relative_transform[2, 3]])


def icp_covariance_calculation(reg_result):
    # TODO
    pass

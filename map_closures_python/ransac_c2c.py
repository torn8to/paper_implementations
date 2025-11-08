import numpy as np
import random
import cv2
from collections import namedtuple

RansacReturnModel = namedtuple("RansacReturnModel", "so2 t theta inliers")


def ransac2d(kpts1: int, kpts2: int, num_iterations: int = 100, inlier_threshold: float = 3.0) -> RansacReturnModel:
    """
    args:
        kpts1: source keypoints
        kpts2: target keypoints
        num_iterrations: the number oif iterations to run ransac over
        num_iterrations: the number oif iterations to run ransac over
    """
    query_pts = np.array(kpts1, dtype=np.float32)
    train_pts = np.array(kpts2, dtype=np.float32)

    best_inliers = []
    best_transform, best_t, best_theta = np.eye(2), np.zeros(2), 0.0

    for _ in range(num_iterations):
        i, j = random.sample(range(len(kpts1)), 2)
        i_pt1, i_pt2 = query_pts[i], train_pts[i]
        j_pt1, j_pt2 = query_pts[j], train_pts[j]

        R, t, theta = estimate_se2_from_set(i_pt1, j_pt1, i_pt2, j_pt2)

        kpts1_est = (R @ query_pts.T).T + t

        errors = np.linalg.norm(kpts1_est - train_pts, axis=1)
        inliers = errors < inlier_threshold

        if np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
            best_inliers = inliers
            best_transform, best_t, best_theta = R, t, theta

    return RansacReturnModel(best_transform, best_t, best_theta, best_inliers)


def estimate_se2_from_set(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray):
    """convert the 2 pairs of points between images to set a transform
    args:
        p1: source point 1
        p2: source point 2
        q1: target point 1
        q2: target point 2

    returns: rotation matrix, positional coordinates and theta

    """
    vp = p2 - p1
    vq = q2 - q1

    vp_norm = vp / np.linalg.norm(vp)
    vq_norm = vq / np.linalg.norm(vq)

    cos_theta = np.dot(vp_norm, vq_norm)
    sin_theta = np.cross(vp_norm, vq_norm)

    theta = np.arctan2(sin_theta, cos_theta)
    R = np.array([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]).reshape((2, 2))

    t = q1 - R @ p1

    return R, t, theta

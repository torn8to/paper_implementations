import numpy as np


def se3_log(T):
    """
    Convert 4x4 SE3 transform matrix to 6x1 twist vector (se3 tangent).
    Order: [vx, vy, vz, wx, wy, wz]
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # rotation angle
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if np.isclose(theta, 0.0):
        # small-angle approximation
        w = np.zeros(3)
        V_inv = np.eye(3)
    else:
        # axis of rotation
        w_hat = (R - R.T) / (2.0 * np.sin(theta))
        w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) * theta

        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta**2)
        V_inv = np.eye(3) - 0.5 * w_hat + (1 - A / (2 * B)) / (theta**2) * (w_hat @ w_hat)

    v = V_inv @ t
    xi = np.concatenate([v, w])
    return xi


def se3_hat(T):
    pass


def get_covariance_matrix(latest_pose: np.ndarray, last_pose: np.ndarray, covariance_uncertainty_multiplier: float) -> np.ndarray:
    """
    Computes the covariance matrix associated with the relative pose between two SE(3) poses used for generating accumulation in pose graph optimization.

    Args:
        latest_pose (np.ndarray): The first pose as a 4x4 transformation matrix.
        last_pose (np.ndarray): The second pose as a 4x4 transformation matrix.

    Returns:
        np.ndarray: and 6x6 covariance matrix
    """
    twist = se3_log(latest_pose @ np.linalg.inv(last_pose))
    return np.diag(twist)

import numpy as np
import scipy


def get_se3_difference(pose1: np.ndarray, pose2: np.ndarray, homogenous_return=true) -> tuple(np.ndarray):
  """
  Computes the relative SE(3) transformation between two poses.

  Args:
      pose1 (np.ndarray): The first pose as a 4x4 transformation matrix.
      pose2 (np.ndarray): The second pose as a 4x4 transformation matrix.
      homogenous_return (bool, optional): If True, returns the result as a homogeneous transformation matrix. Defaults to True.

  Returns:
      tuple(np.ndarray): The relative transformation from pose1 to pose2.
  """
  if homogenous_return == True:
    return np.linalg.inv(pose1) @ pose2
  else:
    p_rel = np.linalg.inv(pose1) @ pose2
    angle = np.arccos((np.trace(R) - 1) / 2.0)
  if abs(angle) < 1e-9:
    axis = np.zeros(3)
  else:
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
  rotvec = axis * angle
  return p_rel[:3, 3], rotvec


def get_covariance_matrix_with_rel_pose(pose1: np.ndarray, pose2: np.ndarray, covariance_uncertainty_multiplier) -> tuple(np.ndarray, np.ndarray):
  """
  Computes the covariance matrix associated with the relative pose between two SE(3) poses.

  Args:
      pose1 (np.ndarray): The first pose as a 4x4 transformation matrix.
      pose2 (np.ndarray): The second pose as a 4x4 transformation matrix.

  Returns:
      tuple(np.ndarray, np.ndarray): A tuple containing thethe relative pose as 4x4 homogenous trnasform matrix and 6 x6 covariance matrix where they think .
  """
  p_rel = get_se3_difference(pose1, pose2)
  xyz, euler = get_se3_difference(pose1, pose2, False)
  np.diag(
    [
      xyz[0] * covariance_uncertainty_multiplier,
      xyz[1] * covariance_uncertainty_multiplier,
      xyz[2] * covariance_uncertainty_multiplier,
      euler[0] * covariance_uncertainty_multiplier,
      euler[1] * covariance_uncertainty_multiplier,
      euler[2] * covariance_uncertainty_multiplier,
    ]
  )

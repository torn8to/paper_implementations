from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import load_config
import numpy as np


"""just a wrapper of the KissICP class for easy access to the point cloud"""


class OdometryWrapper:
  def __init__(self, config_file="./kiss_icp.yaml"):
    self.pipeline = KissICP(load_config(config_file, max_range=100))

  def register_frame(self, frame, timestamps=None):
    if timestamps == None:
      timestamps = np.zeros(frame.shape[0])
    self.pipeline.register_frame(frame, timestamps)

  def get_local_map_cloud(self):
    return self.pipeline.local_map.point_cloud()

  def get_current_position(self):
    return self.pipeline.last_pose

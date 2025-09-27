import pandas as pd
import numpy as np
import os


class Kitti360LidarData:
  def __init__(
    self,
    dataset_dir="~/KITTI-360",
    sequence=0,
    loop=False,
  ):
    sequence_file = f"2013_05_28_drive_00{sequence:02}_sync"
    self.index = 0
    self.num_frames = 0
    self.lidar_dir = os.path.join(dataset_dir, "data_3d_raw", sequence_file, "velodyne_points")
    self.data_dir = os.path.join(self.lidar_dir, "data")
    timestamps_path = os.path.expanduser(os.path.join(self.lidar_dir, "timestamps.txt"))
    self.timestamps = pd.to_datetime(pd.read_csv(os.path.expanduser(timestamps_path), header=None).squeeze("columns"))
    self.num_frames = self.timestamps.shape[0]

    self.image_00 = np.array(
      [
        0.0371783278,
        -0.0986182135,
        0.9944306009,
        1.5752681039,
        0.9992675562,
        -0.0053553387,
        -0.0378902567,
        0.0043914093,
        0.0090621821,
        0.9951109327,
        0.0983468786,
        -0.6500000000,
        0,
        0,
        0,
        1,
      ]
    ).reshape(4, 4)
    self.image_01 = np.array(
      [
        0.0194000864,
        -0.1051529641,
        0.9942668106,
        1.5977241400,
        0.9997374956,
        -0.0100836652,
        -0.0205732716,
        0.5981494900,
        0.0121891942,
        0.9944049345,
        0.1049297370,
        -0.6488433108,
        0,
        0,
        0,
        1,
      ]
    ).reshape(4, 4)
    self.image_02 = np.array(
      [
        0.9995185086,
        0.0041276589,
        -0.0307524527,
        0.7264036936,
        -0.0307926666,
        0.0100608424,
        -0.9994751579,
        -0.1499658517,
        -0.0038160970,
        0.9999408692,
        0.0101830998,
        -1.0686400091,
        0,
        0,
        0,
        1,
      ]
    ).reshape(4, 4)
    self.image_03 = np.array(
      [
        -0.9996821702,
        0.0005703407,
        -0.0252038325,
        0.7016842127,
        -0.0252033830,
        0.0007820814,
        0.9996820384,
        0.7463650950,
        0.0005898709,
        0.9999995315,
        -0.0007674583,
        -1.0751978255,
        0,
        0,
        0,
        1,
      ]
    ).reshape(4, 4)
    self.cam_to_velo = np.array(
      [
        0.04307104361,
        -0.08829286498,
        0.995162929,
        0.8043914418,
        -0.999004371,
        0.007784614041,
        0.04392796942,
        0.2993489574,
        -0.01162548558,
        -0.9960641394,
        -0.08786966659,
        -0.1770225824,
        0,
        0,
        0,
        1,
      ]
    ).reshape(4, 4)
    self.sick_to_velo = np.array(
      [
        0.9998328856,
        -0.01305514558,
        -0.01279702916,
        -0.3971222434,
        0.01322436405,
        0.9998250388,
        0.01322905751,
        -0.009085164561,
        0.0126220829,
        -0.01339607931,
        0.9998305997,
        -0.07072622777,
        0,
        0,
        0,
        1,
      ]
    ).reshape(4, 4)

  def retrieve_next_frame(self) -> np.ndarray:
    frame = self.retrieve_frame_at_index(self.index)
    self.index = self.index + 1
    return frame

  def retrieve_frame_at_index(self, n: int) -> np.ndarray:
    return np.fromfile(os.path.expanduser(os.path.join(self.data_dir, f"{self.index:010}.bin")), dtype=np.float32).reshape((-1, 4))

  def has_next(self) -> bool:
    return self.index < self.num_frames


if __name__ == "__main__":
  loader = Kitti360LidarData()
  loader.retrieve_next_frame()

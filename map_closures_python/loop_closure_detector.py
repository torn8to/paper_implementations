from dataclasses import dataclass, field
from typing import Optional, List, NamedTuple, ClassVar, Iterator
from collections import namedtuple
from itertools import count
from density import generate_density_image_map
from ransac_c2c import ransac2d, RansacReturnModel
from utils import convert_ransac_to_se3_pose
import cv2
import numpy as np

LoopClosureCanidate = namedtuple("LoopClosureCanidate", "id match_id ransac_model matches")


@dataclass
class LoopClosureEntry:
  keypoints: list  # List of OpenCV KeyPoint objects
  descriptors: np.ndarray  # (N, 32) array of ORB descriptors
  img: np.ndarray
  position: np.ndarray = field(default_factory=lambda: np.eye(4))
  cloud:Optional[np.ndarray] = None
  id: int = field(init=False)
  _counter: ClassVar[Iterator[int]] = count(0)
  # _lock: ClassVar[threading.Lock] = threading.Lock()

  def __post_init__(self):  # used to set id post initialization to be set by class variable counter
    self.id = next(type(self)._counter)


@dataclass
class LoopClosureDetectorReturn:
  canidates: list[LoopClosureCanidate]

  def has_valid_canidates(self):
    return len(self.canidates) > 0


class LoopClosureDetector:
  def __init__(self, matched_points_threshold=35, adjacent_id_width_ignore=2):
    self.list_of_loop_closure_entrys: List[LoopClosureEntry] = []
    self.detector_matched_points_threshold = matched_points_threshold
    self.orb = cv2.ORB_create()
    self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)

  def add_new_loop_closure_entry_from_cloud(self, cloud: np.ndarray, position: np.ndarray, resolution=0.5):
    density_img = generate_density_image_map(cloud, position=position[:3, 3], resolution=0.5)
    keypoints, descriptors = self.orb.detectAndCompute(density_img.img, None)
    self.list_of_loop_closure_entrys.append(LoopClosureEntry(keypoints, descriptors, density_img.img, position, cloud))

  def add_new_loop_closure_entry_from_img(self, img: np.ndarray, position: np.ndarray, cloud=None):
    keypoints, descriptors = self.orb.detectAndCompute(img, None)
    self.list_of_loop_closure_entrys.append(LoopClosureEntry(keypoints, descriptors, img, position, cloud))

  def match_and_prune_keypoints(
    self, query_entry: LoopClosureEntry, source_entry: LoopClosureEntry, lowes_ratio: float = 0.7
  ) -> list:
    query_kp = query_entry.keypoints
    query_des = query_entry.descriptors
    source_kp = source_entry.keypoints
    source_des = source_entry.descriptors

    matches = self.matcher.knnMatch(query_des, source_des, k=2)
    good_matches = []
    # TODO: change to actual form this is a placeholder original method does not use lowes ratio
    for i, (m, n) in enumerate(matches):
      if m.distance < lowes_ratio * n.distance:
        good_matches.append([m])
    return query_kp, source_kp, good_matches

  def find_best_possibilities_locations(self, query_entry: LoopClosureEntry, k: int = 1):
    matches = []
    for i in self.list_of_loop_closure_entrys:
      kpts1, kpts2, matches = self.match_and_prune_keypoints(query_entry, i)
      best_model, inliers = ransac2d(kpts1, kpts2, num_iterations=1000)
      if inliers.count_nonzero > self.matched_points_threshold:
        matches.append((i, inliers.count_nonzero()))
    return sorted(matches, key=lambda x: x[1])[:k]

  def process_last_frame(self) -> Optional[LoopClosureDetectorReturn]:
    list_of_loop_closure_canidates: list[LoopClosureCanidate] = []
    if len(self.list_of_loop_closure_entrys) < 5:
      return
    last_frame = self.list_of_loop_closure_entrys[-1]
    for lce in self.list_of_loop_closure_entrys[:-3]:  # skip adjacent frame to last as we already have a connection in the graph
      kpts1, kpts2, matches = self.match_and_prune_keypoints(last_frame, lce)
      
      if len(matches) <= self.detector_matched_points_threshold:
        # shortcut the ransac step if their are not enough points as their is no need to run points if does not meet the matched threshold
        continue
      ransac_model: RansacReturnModel = ransac2d(kpts1, kpts2, matches, num_iterations=1000)
      ransac_pruned_matches: list[bool] = np.asarray(matches, dtype=object)[ransac_model.inliers].tolist()
      if len(matches) > self.detector_matched_points_threshold:
        list_of_loop_closure_canidates.sort(key=lambda x: int(np.count_nonzero(x.ransac_model.inliers)))
        list_of_loop_closure_canidates.append(
          LoopClosureCanidate(last_frame.id, lce.id, ransac_model, matches)
        )
    return LoopClosureDetectorReturn(canidates=list_of_loop_closure_canidates)

  def get_entry_by_id(self, id: int) -> Optional[LoopClosureEntry]:
    for lce in iter(self.list_of_loop_closure_entrys):
      if lce.id == id:
        return lce
    return None

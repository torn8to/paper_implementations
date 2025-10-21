from dataclasses import dataclass, field
from typing import Optional, List, NamedTuple, ClassVar, Iterator
from collections import namedtuple
from itertools import count
from density import generate_density_image_map
from ransac_c2c import ransac2d, RansacReturnModel
from utils import convert_ransac_to_se3_pose, visualize_hbst_match
import cv2
import numpy as np
import pyhbst

LoopClosureCanidate = namedtuple("LoopClosureCanidate", "id match_id ransac_model matches")

class LoopClosureEntry:
  _vertex_id: int = field(init=False)
  pose:np.ndarray = None
  img:np.ndarray = None

@dataclass
class LoopClosureDetectorReturn:
  canidates: list[LoopClosureCanidate]

  def has_valid_canidates(self):
    return len(self.canidates) > 0

class LoopClosureDetector:
  _counter: ClassVar[Iterator[int]] = count(0)
  def __init__(self, max_hamming_distance=45, n_features=1000, split_type = pyhbst.SplitEven):
    self._last_position:Optional[np.ndarray] = None
    self.tree = pyhbst.BinarySearchTree256()
    self._max_hamming_distance = max_hamming_distance
    self.tree_split = split_type
    self.orb = cv2.ORB_create(n_features)

    self.list_of_loop_closure_entrys = []

  def get_loop_closure_entry_by_id(self, id:int):
    return self.list_of_loop_closure_entrys[id]


  def match_and_add_new(self, cloud: np.ndarray, position: np.ndarray, resolution=0.5):
    density_img = generate_density_image_map(cloud, position=position[:3, 3], resolution=0.5)
    id = next(self._counter)
    keypoints, descriptors = self.orb.detectAndCompute(density_img.img, None)
    new_lce = LoopClosureEntry(_vertex_id, pose=position, img=density_img)

    keypoints_list = [key.pt for key in keypoints]
    tree_matches = self.tree.matchAndAdd(keypoints_list, descriptors.tolist(), id, self._max_hamming_distance, self.tree_split)
    top_matches = self.topk_matches(tree_matches, k = id//2 if id > 10 else id)
    non_recent_matches = self.recency_pruning(top_matches, id, 20)

    top_match = None
    if len(non_recent_matches) > 0 and non_recent_matches[0][1] > 2:
            top_match = tree_matches[non_recent_matches[0][0]]
    else:
        return None
    match_reference = non_recent_matches[0][0]
    visualize_top_match(top_match, new_lce.img, self.get_loop_closure_entry_by_id(match_reference)) 

        

  def eval_matches(self, tree_matches):
    pass



  def topk_matches(self, tree_matches, k=50):
    list_of_tm_num = []
    matches_comparator = lambda x: x[1]
    for tm in tree_matches:
      tm_len = len(tree_matches[tm])
      if tm_len > 1:
       list_of_tm_num.append((tm, tm_len))
    list_of_tm_num.sort(key=matches_comparator,reverse=True)
    if k > len(list_of_tm_num):
      return list_of_tm_num
    return list_of_tm_num[:k]

  def recency_pruning(self, matches, id, num_iterations):
    threshold = id-num_iterations
    pruned_matches = []
    for match in matches:
        if match[0] < threshold:
            pruned_matches.append(match)
    return pruned_matches






'''
  def add_odom_check_loop_closure(self) -> Optional[LoopClosureDetectorReturn]:
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

'''

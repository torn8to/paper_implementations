"""Loop closure detection using ORB features on BEV density maps.

This module wraps a binary descriptor index (pyhbst) to match incoming
BEV images against a database and proposes loop-closure candidates.
"""

from dataclasses import dataclass, field
from typing import Optional, List, NamedTuple, ClassVar, Iterator
from collections import namedtuple
from itertools import count
from density import generate_density_image_map
from ransac_c2c import ransac2d, RansacReturnModel
from utils import convert_ransac_to_se3_pose
import cv2
import numpy as np
import pyhbst

LoopClosureCanidate = namedtuple("LoopClosureCanidate", "id match_id ransac_model matches")

@dataclass
class LoopClosureEntry:
    """Single keyframe stored for loop-closure search.

    Attributes
    ----------
    pose : numpy.ndarray of shape (4, 4)
        Pose of the keyframe in the world frame.
    img : numpy.ndarray
        Grayscale BEV density image used for feature extraction.
    """
    _vertex_id: int = field(init=False)
    pose: np.ndarray = None
    img: np.ndarray = None


@dataclass
class LoopClosureDetectorReturn:
    """Return type for loop-closure proposals.

    Attributes
    ----------
    canidates : list of LoopClosureCanidate
        Proposed loop-closure matches and supporting data.
    """
    canidates: list[LoopClosureCanidate]

    def has_valid_canidates(self):
        """Whether at least one candidate is available.

        Returns
        -------
        bool
        """
        return len(self.canidates) > 0


class LoopClosureDetector:
    """Detect loop closures using ORB features indexed in a binary tree.

    Parameters
    ----------
    max_hamming_distance : int, optional
        Maximum allowed Hamming distance for descriptor matches.
    n_features : int, optional
        Number of ORB features to detect per image.
    split_type : pyhbst.Split*, optional
        Split strategy for the binary search tree.

    Attributes
    ----------
    tree : pyhbst.BinarySearchTree256
        Descriptor index used for approximate matching.
    orb : cv2.ORB
        ORB detector/descriptor instance.
    _max_hamming_distance : int
    list_of_loop_closure_entrys : list[LoopClosureEntry]
    """
    _counter: ClassVar[Iterator[int]] = count(0)

    def __init__(self, max_hamming_distance=35, n_features=500, split_type=pyhbst.SplitEven):
        self._last_position: Optional[np.ndarray] = None
        self.tree = pyhbst.BinarySearchTree256()
        self._max_hamming_distance = max_hamming_distance
        self.tree_split = split_type
        self.orb = cv2.ORB_create(n_features)

        self.list_of_loop_closure_entrys = []

    def get_loop_closure_entry_by_id(self, id: int):
        """Retrieve a stored entry by its index.

        Parameters
        ----------
        id : int
            Identifier assigned to the entry when added to the index.

        Returns
        -------
        LoopClosureEntry
        """
        return self.list_of_loop_closure_entrys[id]

    def match_and_add_new(self, density_img, position: np.ndarray, id):
        """Extract features, match against the database, and add the new frame.

        Parameters
        ----------
        density_img : DensityMap or object with attribute ``img`` (numpy.ndarray)
            BEV density image for the current frame.
        position : numpy.ndarray of shape (4, 4)
            Pose of the current frame in the world frame. (Currently unused.)
        id : int
            Unique identifier for the current frame.

        Returns
        -------
        tuple | None
            ``(id, matches)`` for the best prior frame, where ``matches`` is the
            raw match list from the tree, or ``None`` if no suitable match.
        """
        keypoints, descriptors = self.orb.detectAndCompute(density_img.img, None)
        keypoints_list = [key.pt for key in keypoints]
        tree_matches = self.tree.matchAndAdd(keypoints_list, descriptors.tolist(), id, self._max_hamming_distance, self.tree_split)
        top_matches = self.topk_matches(tree_matches, k=id // 2 if id > 10 else id)
        non_recent_matches = self.recency_pruning(top_matches, id, 20)
        if len(non_recent_matches) > 5:
            print(non_recent_matches[:5])
        else:
            print(non_recent_matches)
        top_match = None
        if len(non_recent_matches) > 0 and non_recent_matches[0][1] > 25:
            top_match = tree_matches[non_recent_matches[0][0]]
        else:
            return None
        return (id, top_match)

    def eval_matches(self, tree_matches):
        """Evaluate match sets and compute an overall score.

        Parameters
        ----------
        tree_matches : dict[int, list]
            Mapping from prior frame id to list of descriptor matches.

        Returns
        -------
        Any
            Not implemented.
        """
        pass

    def topk_matches(self, tree_matches, k=50):
        """Select top-k prior frames by number of descriptor matches.

        Parameters
        ----------
        tree_matches : dict[int, list]
            Mapping from prior frame id to list of matches.
        k : int, default: 50
            Number of top entries to return.

        Returns
        -------
        list[tuple[int, int]]
            List of ``(frame_id, count)`` sorted by count descending.
        """
        list_of_tm_num = []
        matches_comparator = lambda x: x[1]
        for tm in tree_matches:
            tm_len = len(tree_matches[tm])
            if tm_len > 1:
                list_of_tm_num.append((tm, tm_len))
        list_of_tm_num.sort(key=matches_comparator, reverse=True)
        if k > len(list_of_tm_num):
            return list_of_tm_num
        return list_of_tm_num[:k]

    def recency_pruning(self, matches, id, num_iterations):
        """Filter out matches that are too recent.

        Parameters
        ----------
        matches : list[tuple[int, int]]
            List of candidate matches ``(frame_id, count)``.
        id : int
            Current frame id.
        num_iterations : int
            Minimum separation from current id.

        Returns
        -------
        list[tuple[int, int]]
            Pruned list with ``frame_id < id - num_iterations``.
        """
        threshold = id - num_iterations
        pruned_matches = []
        for match in matches:
            if match[0] < threshold:
                pruned_matches.append(match)
        return pruned_matches


"""
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

"""

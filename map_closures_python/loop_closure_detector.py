"""Loop closure detection using ORB features on BEV density maps.

This module wraps a binary descriptor index (pyhbst) to match incoming
BEV images against a database and proposes loop-closure candidates.
"""

from dataclasses import dataclass, field
from typing import Optional, List, ClassVar, Iterator, Callable
from collections import namedtuple
from itertools import count
from density import generate_density_image_map
from ransac_c2c import ransac2d, RansacReturnModel
from utils import convert_ransac_to_se3_pose
import cv2
import numpy as np
import pyhbst
from copy import deepcopy

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
    max_hamming_distance : int
    list_of_loop_closure_entrys : list[LoopClosureEntry]
    """
    _counter: ClassVar[Iterator[int]] = count(0)

    def __init__(self,
                 resolution:float = 0.5,
                 max_hamming_distance=35,
                 n_features=500,
                 split_type=pyhbst.SplitEven,
                 ransac_inlier_threshold: float = 3.0,
                 match_criteria_threshold: int = 25,
                 post_ransac_threshold: int = 25,
                 distance_threshold=15):
        self.bev_resolution = resolution
        self.distance_threshold = distance_threshold
        self._last_position: Optional[np.ndarray] = None
        self.tree = pyhbst.BinarySearchTree256()
        self.max_hamming_distance = max_hamming_distance
        self.tree_split = split_type
        self.orb = cv2.ORB_create(n_features)
        self.inlier_threshold = ransac_inlier_threshold
        self.ransac_threshold = match_criteria_threshold
        self.post_ransac_threshold = post_ransac_threshold
        self.list_of_loop_closure_entrys:list[tuple] = []

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


    def extract_keypoints_from_match(self,match_obj):
        query = []
        source = []
        for i in range(len(match_obj)):
            query.append(match_obj[i].object_query)
            source.append(match_obj[i].object_references[0])
        return (query, source)
    

    def match_and_add_new(self, density_img, position: np.ndarray, id)->Optional[tuple[int,RansacReturnModel,]]:
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
        distance_pruning: Callable[tuple[int, result, np.ndarray], bool] = lambda x: np.linalg.norm(x[1].t) > self.distance_threshold
        keypoints, descriptors = self.orb.detectAndCompute(density_img.img, None)
        keypoints_list = [key.pt for key in keypoints]
        tree_matches = self.tree.matchAndAdd(keypoints_list, descriptors.tolist(), id, self.max_hamming_distance, self.tree_split)
        top_matches = self.topk_matches(tree_matches, k=id // 2 if id > 10 else id)
        non_recent_matches = self.recency_pruning(top_matches, id, 20)
        
        if len(non_recent_matches) == 0:
            return None
        ransac_list = []
        
        for i in non_recent_matches:
            query_id = i[0]
            match = tree_matches[i[0]]
            query_pts, source_pts = self.extract_keypoints_from_match(match)
            result = ransac2d(query_pts, source_pts,inlier_threshold=self.ransac_threshold)
            ransac_list.append((query_id, result, convert_ransac_to_se3_pose(result, self.bev_resolution)))

        conditional: Callable[tuple[int, result], bool] = lambda x: len(x[1].inliers) > self.post_ransac_threshold
        inlier_filtered_list = [i for i in ransac_list if conditional(i)]
        distance_filtered_list = [i for i in inlier_filtered_list if distance_pruning(i)]
        print(f"inlier_filtered length {len(inlier_filtered_list)} distance_filtered {len(distance_filtered_list)}")
        return distance_filtered_list

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
            if tm_len > 1: # minimum threshold for matches to build 
                list_of_tm_num.append((tm, tm_len))
        list_of_tm_num.sort(key=matches_comparator, reverse=True)
        if k > len(list_of_tm_num):
            return list_of_tm_num
        return list_of_tm_num[:k]

    def recency_pruning(self, matches, id, num_iterations=200):
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

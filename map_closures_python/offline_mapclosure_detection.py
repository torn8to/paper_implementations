from loop_closure_detector import LoopClosureEntry, LoopClosureDetector, LoopClosureDetectorReturn, LoopClosureCanidate
from pgo import PGO
from lie_algebra import get_covariance_matrix
from utils import plot_transformation_matrices, convert_ransac_to_se3_pose
import numpy as np
from registration import registrationICP
import matplotlib.pyplot as plt
import pickle
import timeit

list_imgs_pos = pickle.load(open("loop_closure_data_xl.pkl","rb"))
detection_matching_threshold = 15
lcd = LoopClosureDetector()
pgo = PGO()

matched_frames = None
list_of_matched_frames = []
last_pose = np.eye(4)


for i in range(len(list_imgs_pos) - 1):
    lcd.add_new_loop_closure_entry_from_img(list_imgs_pos[i][0],
                                            list_imgs_pos[i][1],
                                            list_imgs_pos[i][2])
    covariance = get_covariance_matrix(list_imgs_pos[i][1], last_pose, 0.02)
    pgo.add_odom(list_imgs_pos[i][1], last_pose, covariance)
    update_covariance = covariance(list_imgs_pos[i])
    matched_frames: LoopClosureDetectorReturn = lcd.process_last_frame()
    lambda: lcd.process_last_frame()
    if matched_frames != None and matched_frames.has_valid_canidates():
        for match in matched_frames:
            se3_prior = convert_ransac_to_se3_pose(match.ransac, 0.5)
            match_norm = np.linalg.norm(se3_prior[3,:3])
            if match_norm < detection_matching_threshold():
                source_cloud = lcd.get_entry_by_id(match.id).cloud,
                target_cloud = lcd.get_entry_by_id(match.match_id).cloud,
                final_transform = registrationICP(
                    source_cloud,
                    target_cloud,
                    se3_prior)
                pgo.add_loop_closure_edge(match.id,
                                          match.match_id,
                                          final_transform)



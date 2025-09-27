from loop_closure_detector import LoopClosureEntry, LoopClosureDetector, LoopClosureDetectorReturn, LoopClosureCanidate
from pgo import SimplePGO
from utils import plot_transformation_matrices
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ransac_c2c import ransac2d
import pickle
import timeit

list_imgs_pos = pickle.load(open("loop_closure_data_xl.pkl","rb"))

lcd = LoopClosureDetector()
pose_graph_optimizer = SimplePGO()
matched_frames = None
list_of_matched_frames = []
for i in range(len(list_imgs_pos) - 1):
    lcd.add_new_loop_closure_entry_from_img(list_imgs_pos[i][0], list_imgs_pos[i][1])
    matched_frames: LoopClosureDetectorReturn = lcd.process_last_frame()
    t = timeit.timeit(lambda: lcd.process_last_frame(),number=1)
    print(f"frame {i} processed at time {t}")
    if matched_frames != None and matched_frames.has_valid_canidates():
        list_of_matched_frames.append(matched_frames)

transforms = [pair[1] for pair in list_imgs_pos]
connections:list[tuple] = []
for match in list_of_matched_frames:
    for canidate in match.canidates:
        connections.append((canidate.id,canidate.match_id))
print(len(list_of_matched_frames),len(connections))
fig = plot_transformation_matrices(transforms, connections=connections, 
                                      title="Loop Detection pairs on an LoopClosureDetected Pairs with a threshold of 35")
plt.show()

'''
print(len(matched_frames.canidates))
top_canidate: LoopClosureCanidate = matched_frames.canidates[0]
query: LoopClosureEntry = lcd.get_entry_by_id(top_canidate.id) 
source: LoopClosureEntry = lcd.get_entry_by_id(top_canidate.match_id) 
print(type(query.keypoints), type(query.keypoints[0]))
kpts1 = query.keypoints
kpts2 = source.keypoints
ransac_pruned_matches: list[bool] = np.asarray(top_canidate.matches, dtype=object)[top_canidate.ransac_model.inliers].tolist()
ransac_match_img = cv2.drawMatchesKnn(
  127 - query.img, kpts1,
  127 - source.img, kpts2,
  ransac_pruned_matches, None,
  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.imshow(ransac_match_img)
plt.title("loop closure detection")
plt.show()
'''




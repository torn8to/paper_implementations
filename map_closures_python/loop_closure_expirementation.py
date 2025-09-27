from loop_closure_detector import LoopClosureEntry, LoopClosureDetector
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ransac_c2c import ransac2d
import pickle

list_imgs_pos = pickle.load(open("loop_closure_data_xl.pkl", "rb"))

print(len(list_imgs_pos))

lcd = LoopClosureDetector()
for tup in list_imgs_pos:
  lcd.add_new_loop_closure_entry_from_img(tup[0], tup[1])
list_lce = lcd.list_of_loop_closure_entrys

last_entry = list_lce[-1]
slast_entry = list_lce[-2]
kpts1, kpts2, matches = lcd.match_and_prune_keypoints(last_entry, slast_entry)


print(f"Number of keypoints 1: {len(kpts1)}")
print(f"Number of keypoints 2: {len(kpts2)}")
print(f"Number of matches: {len(matches)}")

# invert images
match_img = cv2.drawMatchesKnn(
  127 - last_entry.img, kpts1, 127 - slast_entry.img, kpts2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


ransac_out = ransac2d(kpts1, kpts2, matches, num_iterations=1000)
ransac_pruned_matches = np.asarray(matches, dtype=object)[ransac_out.inliers].tolist()

print(f"ransac pruned matches {len(ransac_pruned_matches)}")
print(f"best_model: {ransac_out.so2} t:{ransac_out.t}")

ransac_match_img = cv2.drawMatchesKnn(
  127 - last_entry.img, kpts1,
  127 - slast_entry.img, kpts2,
  ransac_pruned_matches, None,
  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
fig, ax = plt.subplots(2)
fig.suptitle("matching stages with pruning using orb with distance pruning strategy and ransac")
ax[0].set_title("orb descriptor matching after lowes ratio test")
ax[0].imshow(match_img)
ax[1].set_title("orb descriptor matching after ransac")
ax[1].imshow(ransac_match_img)
plt.show()

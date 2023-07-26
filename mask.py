import numpy as np
import pickle
import matplotlib.pyplot as plt

save_path = './data/common/'


m = np.ones((9, 22))
name = 'mask_thumb_right.p'
m[:, 11:13] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_index_right.p'
m[:5, 13:16] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_middle_right.p'
m[:5, 16:18] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_ring_right.p'
m[:5, 18:20] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_little_right.p'
m[:5, 20:22] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_palm_right.p'
m[5:, 13:] = 0 
pickle.dump(m, open(save_path + name, "wb"))


m = np.ones((9, 22))
name = 'mask_thumb_left.p'
m[:, 8:11] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_index_left.p'
m[:5, 6:8] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_middle_left.p'
m[:5, 4:6] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_ring_left.p'
m[:5, 2:4] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_little_left.p'
m[:5, 0:2] = 0 
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_palm_left.p'
m[5:, :8] = 0 
pickle.dump(m, open(save_path + name, "wb"))


m = np.ones((9, 22))
name = 'mask_fingertips_right.p'
m[:3, 11:] = 0
m[:4, 11:13] = 0
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_fingers_right.p' # all fingers
m[:5, 11:] = 0
m[:, 11:13] = 0
pickle.dump(m, open(save_path + name, "wb"))

m = np.ones((9, 22))
name = 'mask_fingermids_right.p'
m[3:5, 13:] = 0
m[4:5, 11:13] = 0
pickle.dump(m, open(save_path + name, "wb"))
 

m = np.ones((9, 22))
name = 'mask_mid_right.p'
m[4:6, 11:] = 0
pickle.dump(m, open(save_path + name, "wb"))
 
 
 
# plt.imshow(m)
# plt.show()


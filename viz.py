import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


main_path = './dataset/01_bolt/predictions/'
data = pickle.load(open(main_path + 'train_01_bolt.p', "rb")) # train
# data = pickle.load(open(main_path + 'train_best_01_bolt.p', "rb")) # val 
# data = pickle.load(open(main_path + 'eval_01_bolt.p', "rb")) # test


# tac_left = data[0]
# tac_right = data[1]
tactile = data[0]
gt = data[1]
pred = data[2]

pred = softmax(pred, axis=1)

fig, ax = plt.subplots(1,2) 

ax[0].imshow(gt,  aspect=0.1)
ax[1].imshow(pred,  aspect=0.1)
plt.show()

# for i in range(tactile.shape[0]):
#     plt.imshow(tactile[i, 0, :, :])
#     print ('pred stage:', np.argmax(pred[i, :]), 'gt stage:', np.argmax(gt[i, :]))
#     plt.show()
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


main_path = './dataset/0622/predictions/'
# data = pickle.load(open(main_path + 'train_0622_test.p', "rb"))
data = pickle.load(open(main_path + 'train_best_0622_test.p', "rb"))

tac_left = data[0]
tac_right = data[1]
gt = data[2]
pred = data[3]

pred = softmax(pred, axis=1)

fig, ax = plt.subplots(1,2) 

ax[0].imshow(gt)
ax[1].imshow(pred)
plt.show()


import cv2
import numpy as np
import pandas as pd
from utils import *
import matplotlib as mpl
import h5py

# main_path = './data/processed/01_bolt/'
main_path = './data/'
name = '0724_rec01_dyn'

tac_left = pickle.load(open(main_path + name + '_left.p', 'rb'))
tac_right = pickle.load(open(main_path + name + '_right.p', 'rb'))

for i in range(3846, 15000):
    print (i)
    plt.imshow(tac_left[i, :, :], vmin=0, vmax=0.5, cmap='binary')
    plt.axis('off')
    plt.show()
    plt.imshow(tac_right[i, :, :], vmin=0, vmax=0.5, cmap='binary')
    plt.axis('off')
    plt.show()


import numpy as np
import serial
import time
import h5py
import datetime
import matplotlib.pyplot as plt

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

path = './data/0725_rec01_dyn_scale.hdf5'
f = h5py.File(path, 'r')
fc = f['frame_count'][0]
ts = np.array(f['ts'][:fc])
scale = np.array(f['scale'][:fc]).astype(np.float32)

scale = np.where(scale<1e6, 0, scale)
n = np.min(scale)
while n==0:
    for i in range(1, len(scale)):
        if scale[i] == 0:
            scale[i] = scale[i-1]

    for i in range(len(scale)-1):
        if scale[i] == 0:
            scale[i] = scale[i+1]
          
    n = np.min(scale)

scale = smooth(scale, 2)

print (fc, ts, scale)
plt.plot(scale)
plt.show()
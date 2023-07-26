import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


def window_select(data, timestep, window):
    if window ==0:
        return data[0][timestep:timestep+1, :, :], data[1][timestep:timestep+1, :, :], data[2][timestep:timestep+1, :] 

    lo = max(0 , timestep-window)
    if lo == 0:
        return data[0][:window,:,:], data[1][:window,:,:], data[2][timestep:timestep+1,:]
    else:
        return data[0][timestep-window:timestep,:,:], data[1][timestep-window:timestep,:,:], data[2][timestep:timestep+1,:]

def get_subsample(tac, subsample):
    for x in range(0, tac.shape[1], subsample):
        for y in range(0, tac.shape[2], subsample):
            v = np.mean(tac[:, x:x+subsample, y:y+subsample], (1, 2))
            tac[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)

    return tac

class sample_data(Dataset):
    def __init__(self, path, window, subsample):
        self.path = path
        self.log = np.asanyarray(pickle.load(open(self.path + 'log.p', "rb")))
        # print (self.log)
        self.window = window
        self.subsample = subsample

    def __len__(self):
        # return 4
        return self.log[-1]

    def __getitem__(self, idx):
        f = np.where(self.log<=idx)[0][-1]
        local_lo = self.log[f]
        local_hi = self.log[f+1]

        local_idx = idx - local_lo

        local_path = self.path + str(self.log[f]) + '.p'
        data = pickle.load(open(local_path, "rb"))

        # data[2] = np.reshape(data[2], (1, -1))

        tac_left, tac_right, label = window_select(data, local_idx, self.window)
        tac = np.concatenate((tac_left, tac_right), axis=2)

        if self.subsample > 1:
            tac = get_subsample(tac, self.subsample)

        return tac, label
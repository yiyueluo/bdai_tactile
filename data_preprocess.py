import scipy.io
import numpy as np
import pickle
from utils import *

date = '0622'
name = 'rec01'
st_frame = 50  # needs to be modified accordingly
phase = ['train', 'val']
ratio = 0.8
main_path = './data/'
dump_path = './dataset/' + date + '/'

# load labels from .mat file
mat = scipy.io.loadmat(main_path + 'videoLabelingSession_' + date + '_' + name + '_arr.mat')
label_data = mat['LabelData']
print ('labels loaded:', mat['LabelData'].shape)

# load normalized and spatially aligned tactile data
tac_left = pickle.load(open(main_path + date + '_' + name + '_left.p', 'rb'))
tac_right = pickle.load(open(main_path + date + '_' + name + '_right.p', 'rb'))
print ('tactile loaded', tac_left.shape, tac_right.shape)

n_label = label_data.shape[1]
n_frames = tac_left.shape[0]

com_arr = np.zeros((st_frame, label_data.shape[1]))
label_data = np.concatenate((com_arr, label_data), axis=0) # added missed frames to the label data at the begining

# extract stamps from .mat file
extracted_fs = []
for nl in range(n_label):
    extracted_fs.append(np.where(label_data[:, nl]==1))

n_stages = len(extracted_fs) - 1
n_rounds = extracted_fs[0][0].shape[0]

print ('n_stages:', n_stages, 'n_rounds:', n_rounds)
# print (extracted_fs)


# split to train, val, export dataset
seq = [[0, np.int16(ratio*n_rounds)], [np.int16(ratio*n_rounds), n_rounds]]

for p in range(len(phase)):
    count = [0]
    c = 0
    for nr in range(seq[p][0], seq[p][1]):
        seq_st = extracted_fs[0][0][nr]
        seq_ed = extracted_fs[-1][0][nr]
        aligned_label = np.zeros((seq_ed-seq_st, n_stages))
        aligned_tactile_left = tac_left[seq_st:seq_ed, :, :]
        aligned_tactile_right = tac_right[seq_st:seq_ed, :, :]
        for nl in range(n_stages):
            st = extracted_fs[nl][0][nr] - seq_st
            ed = extracted_fs[nl+1][0][nr] - seq_st
            aligned_label[st:ed, nl] = 1
            # print (st, ed)

        # subst = extracted_fs[0][0][nr] - seq_st
        # subed = extracted_fs[2][0][nr] - seq_st
        # to_save = [aligned_tactile_left[subst:subed, :, :], aligned_tactile_right[subst:subed, :, :], aligned_label[subst:subed, :2]]
        to_save = [aligned_tactile_left, aligned_tactile_right, aligned_label]
        save_path = dump_path + phase[p] + '/' + str(count[-1]) + '.p'
        pickle.dump(to_save, open(save_path, "wb"))
        print ('dumped', save_path, to_save[0].shape, to_save[1].shape, to_save[2].shape)
        print (np.amax(to_save[2]))
        c += to_save[0].shape[0]
        count.append(c)

        # viz to check
        # plt.imshow(to_save[2], aspect=0.01)
        plt.imshow(to_save[0][20], vmin=0, vmax=0.4)
        plt.show()
        plt.imshow(to_save[1][20], vmin=0, vmax=0.4)
        plt.show()


    pickle.dump(count, open(dump_path + phase[p] + '/log.p', "wb"))
    print (phase[p], 'dataset exported.', "n_frames:", count)



# align labels
# aligned_label = np.zeros((n_frames, n_label))
# for nr in range(n_rounds):
#     for nl in range(n_stages):
#         st = extracted_fs[nl][0][nr]
#         ed = extracted_fs[nl+1][0][nr]
#         aligned_label[st:ed, nl] = 1



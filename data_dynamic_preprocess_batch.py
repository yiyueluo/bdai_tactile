import scipy.io
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import *



def load_files(tac_paths, label_path, st_frame):
    tac_left = pickle.load(open(tac_paths[0], 'rb'))
    tac_right = pickle.load(open(tac_paths[1], 'rb'))
    # print ('tactile loaded', tac_left.shape, tac_right.shape)

    mat = scipy.io.loadmat(label_path)
    label_data = mat['LabelData']
    # print ('labels loaded:', mat['LabelData'].shape)

    # added missed frames to the label data at the begining
    com_arr = np.zeros((st_frame, label_data.shape[1]))
    label_data = np.concatenate((com_arr, label_data), axis=0)

    print ('data loaded', tac_left.shape, tac_right.shape, label_data.shape) # nf x 9 x 11; nf x ncls
    if (label_data.shape[0] != tac_left.shape[0]):
        print ('check frame counts!!!')
        print(label_data.shape[0], tac_left.shape[0])

    return tac_left, tac_right, label_data


def split_by_mode(tac_left, tac_right, label_data, sted, dyn_data, mode_count):
    extracted_st = (np.where(label_data[:, sted[0]]==1))[0]
    extracted_ed = (np.where(label_data[:, sted[1]]==1))[0]

    if extracted_st.shape[0] != extracted_ed.shape[0]:
        print ("missing time stamp!!!")

    if extracted_st.shape[0] != sum(mode_count):
        print ("wrong round numbers!!!")
        print (extracted_st.shape[0], sum(mode_count))

    tac_left_by_mode =[]
    tac_right_by_mode =[]
    dyn_by_mode = []
    count = 0
    for m in range(len(mode_count)):
        count += mode_count[m]
        tac_left_by_mode.append([])
        tac_right_by_mode.append([])
        dyn_by_mode.append([])
        for n in range(count-mode_count[m], count):
            tac_left_by_mode[m].append(tac_left[extracted_st[n]:extracted_ed[n], :, :])
            tac_right_by_mode[m].append(tac_right[extracted_st[n]:extracted_ed[n], :, :])
            dyn_by_mode[m].append(dyn_data[extracted_st[n]:extracted_ed[n]])

    return tac_left_by_mode, tac_right_by_mode, dyn_by_mode # [[arr1, arr2...], [], []]


def export(tac_left, tac_right, dyn_data, export_count, phase, ratio, save_path):
    if (len(tac_left) != len(dyn_data[0])) or (len(tac_left) != sum(ratio)):
        print ('check data length!!!')
        print (len(tac_left), len(dyn_data[0]), sum(ratio))

    count = 0
    for p in range(len(phase)):
        count += ratio[p]
        for n in range(count-ratio[p], count):
            # remove incorrect insertion label
            to_save = [tac_left[n], tac_right[n]]
            for i in range(len(dyn_data)):
                to_save.append(np.reshape(dyn_data[i][n], (-1, 1)))
            path = save_path + phase[p] + '/' + str(export_count[p][-1]) + '.p'
            pickle.dump(to_save, open(path, "wb"))
            print ('dumped', save_path, to_save[0].shape, to_save[1].shape, to_save[2].shape)
            export_count[p].append(export_count[p][-1] + to_save[0].shape[0])

    return export_count



task = '_03'
name = '03_case'
mode = 1 # free, dominant, non-dominant, 2-hand
viz = True
scale_bool = False
# dyn_para_name = 'mocap' # mocap / scale

main_path = './data/processed/' + name + '/'
n_mode = 4 # number of data collection mode
phase = ['train', 'val', 'test'] # 7:2:1
export_count = [[0], [0], [0]]
save_path = './dataset/' + name + '_dyn/'


# read log file
df = pd.read_csv(main_path + 'log.csv', sep=',', header=0)
log = df.to_numpy()
n_file = log.shape[0]

for f in range(n_file):
    # extract meta data
    date = str("{:04d}".format(log[f, 0]))
    p = str("{:02d}".format(log[f, 1]))
    p += task
    if log[f, 2] != 0:
        p += '_'
        p += str(log[f, 2])
    
    # load files
    pp = str("{:02d}".format(log[f, 1])) + '_dyn'
    tac_left_path = main_path + 'rec' + pp + '_left.p'
    tac_right_path = main_path + 'rec' + pp + '_right.p'
    tac_paths = [tac_left_path, tac_right_path]
    label_path = main_path + 'rec' + p + '_arr.mat'
    st_frame = log[f, 3 + n_mode]

    tac_left, tac_right, label_data = load_files(tac_paths, label_path, st_frame)

    if name == '03_case':
        dyn_data = pickle.load(open(main_path + 'rec' + pp + '_mocap.p', 'rb'))
        dyn_data = dyn_data[0][0]
        sted = [0, 4] # set the start and end label
    elif name == '04_wrench':
        dyn_data = pickle.load(open(main_path + 'rec' + pp + '_mocap.p', 'rb'))
        dyn_data = dyn_data[3][2]
        sted = [1, 5] # set the start and end label
    elif name == '05_pitch_right':
        dyn_data = pickle.load(open(main_path + 'rec' + pp + '_mocap.p', 'rb'))
        dyn_data = dyn_data[4][2]
        dyn_data2 = pickle.load(open(main_path + 'rec' + pp + '_scale.p', 'rb'))
        sted = [1, 4] # set the start and end label
    elif name == '06_pitch_left':
        dyn_data = pickle.load(open(main_path + 'rec' + pp + '_mocap.p', 'rb'))
        dyn_data = dyn_data[5][2]
        sted = [1, 4] # set the start and end label
        dyn_data2 = pickle.load(open(main_path + 'rec' + pp + '_scale.p', 'rb'))

    print ('dyn data loaded', dyn_data.shape)

    mode_count = []
    for nn in range(n_mode):
        mode_count.append(log[f, 3+nn]) # free, dominant, non-dominant, 2-hand

    # split by mode
    tac_left_by_mode, tac_right_by_mode, dyn_by_mode = split_by_mode(tac_left, tac_right, label_data, sted, dyn_data, mode_count)
    tac_left = tac_left_by_mode[mode]
    tac_right = tac_right_by_mode[mode]
    dyn_data = dyn_by_mode[mode]

    dyn = [dyn_data]

    if name == '05_pitch_right' or name == '06_pitch_left':
        _, _, dyn2_by_mode = split_by_mode(tac_left, tac_right, label_data, sted, dyn_data2, mode_count)
        dyn_data2 = dyn2_by_mode[mode]
        dyn.append(dyn_data2)

    # dump as train/val/test set
    ratio = [log[f, 4 + n_mode], log[f, 5 + n_mode], log[f, 6 + n_mode]]
    export_count = export(tac_left, tac_right, dyn, export_count, phase, ratio, save_path)

    if viz:
        plt.plot(dyn_data[0])
        plt.show()

# dump log for each phase
for p in range(len(phase)):
    if export_count[p][-1] != 0:
        pickle.dump(export_count[p], open(save_path + phase[p] + '/log.p', "wb"))
        print (phase[p], export_count[p])

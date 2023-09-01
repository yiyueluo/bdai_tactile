import scipy.io
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import os
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
    if label_data.shape[0] != tac_left.shape[0]:
        print ('check frame counts!!!')
        print(label_data.shape[0], tac_left.shape[0])

    return tac_left, tac_right, label_data


def split_by_mode(tac_left, tac_right, label_data, mode_count):
    extracted_st = (np.where(label_data[:, 0]==1))[0]
    extracted_ed = (np.where(label_data[:, -1]==1))[0]
    # print ('tac', extracted_st, extracted_ed)

    if extracted_st.shape[0] != extracted_ed.shape[0]:
        print ("missing time stamp!!!")
        print (extracted_st.shape[0], extracted_ed.shape[0])

    if extracted_st.shape[0] != sum(mode_count):
        print ("wrong round numbers!!!")
        print (extracted_st.shape[0], sum(mode_count))

    label_by_mode = []
    tac_left_by_mode =[]
    tac_right_by_mode =[]
    count = 0
    for m in range(len(mode_count)):
        count += mode_count[m]
        label_by_mode.append([])
        tac_left_by_mode.append([])
        tac_right_by_mode.append([])
        for n in range(count-mode_count[m], count):
            label_by_mode[m].append(label_data[extracted_st[n]:extracted_ed[n], :])
            tac_left_by_mode[m].append(tac_left[extracted_st[n]:extracted_ed[n], :, :])
            tac_right_by_mode[m].append(tac_right[extracted_st[n]:extracted_ed[n], :, :])
            # print ('tac', m, n, extracted_ed[n], extracted_st[n], tac_right_by_mode[m][-1].shape)

    return tac_left_by_mode, tac_right_by_mode, label_by_mode # [[arr1, arr2...], [], []]


def split_dyn_by_mode(tac_left, tac_right, label_data, sted, dyn_data, name_list, nt, mode_count):
    extracted_st = (np.where(label_data[:, 0]==1))[0]
    extracted_ed = (np.where(label_data[:, -1]==1))[0]
    # print ('dyn', extracted_st, extracted_ed)

    extracted_st_dyn = (np.where(label_data[:, sted[0]]==1))[0]
    extracted_ed_dyn = (np.where(label_data[:, sted[1]]==1))[0]

    # print (len(extracted_st_dyn), len(extracted_ed_dyn), len(extracted_st), len(extracted_ed))

    if extracted_st.shape[0] != extracted_ed.shape[0]:
        print ("missing time stamp!!!")

    if extracted_st.shape[0] != sum(mode_count):
        print ("wrong round numbers!!!")
        print (extracted_st.shape[0], sum(mode_count))

    dyn_by_mode = []
    dyn_mask_by_mode = []
    count = 0
    for m in range(len(mode_count)):
        count += mode_count[m]
        dyn_by_mode.append([])
        dyn_mask_by_mode.append([])
        for n in range(count-mode_count[m], count):   
            mask = np.zeros((extracted_ed[n] - extracted_st[n], len(name_list))) # update number of tasks
            mask[extracted_st_dyn[n]- extracted_st[n]:extracted_ed_dyn[n]- extracted_st[n], nt] = 1 # remove seq without menaingful dynamic param
            # to_append = dyn_data[extracted_st[n]:extracted_ed[n]] * comp
            dyn_by_mode[m].append(dyn_data[extracted_st[n]:extracted_ed[n]])
            dyn_mask_by_mode[m].append(mask)
            # print ('dyn', m, n, extracted_ed[n], extracted_st[n], dyn_by_mode[m][-1].shape, dyn_mask_by_mode[m][-1].shape)

    return dyn_by_mode, dyn_mask_by_mode # [[arr1, arr2...], [], []]


def split_by_stage(label_data, label_count_list, nt, name_list): #[arr1, arr2 ...]
    n_stages = label_data[0].shape[1]
    n_rounds = len(label_data)

    label = []
    label_mask = []
    for r in range(n_rounds):
        tl = tac_left[r] # arr
        tr = tac_right[r]
        l = label_data[r]
        full_loutput = np.zeros((l.shape[0], sum(label_count_list)))
        louput = np.zeros((l.shape[0], l.shape[1]))# [arr1, arr2 ...]

        extracted_fs = [0]
        for s in range(1, n_stages-1):
            fs = np.where(l[:, s]==1)[0]
            if len(fs) == 0:
                fs = [np.inf]
            # print (fs)
            extracted_fs.append(fs[0])
        extracted_fs.append(l.shape[0])

        extracted_fs_ori = extracted_fs.copy()
        extracted_fs.sort() #sorted time stamp

        st_index =0
        ed = 0
        while ed < l.shape[0]:
            st = extracted_fs[st_index]
            ed = extracted_fs[st_index+1] 
            label_index = extracted_fs_ori.index(extracted_fs[st_index+1]) 
            louput[st:ed, label_index] = 1
            st_index += 1
        
        # remove incorrect action labels
        if name_list[nt] == '01_bolt':
            louput = np.delete(louput, [3, 4], 1)
        if name_list[nt] == '02_screw':
            louput = np.delete(louput, [3], 1)
        if name_list[nt] == '03_case':
            louput = np.delete(louput, [3], 1)

        # remove first label becuase n_stages = n_label - 1
        full_loutput[:, sum(label_count_list[:nt]):sum(label_count_list[:nt])+label_count_list[nt]] = louput[:, 1:]

        # generate label mask
        moutput = np.ones((full_loutput.shape[0], full_loutput.shape[1]))

        for m in range(moutput.shape[0]):
            if np.sum(full_loutput[m, :]) == 0:
                moutput[m, :] = 0
        for m in range(moutput.shape[1]):
            if np.sum(full_loutput[:, m]) == 0:
                moutput[:, m] = 0
        
        label.append(full_loutput) 
        label_mask.append(moutput)
    
    return label, label_mask  # [arr1, arr2, ...]


def export(name_list, nt, contrain, tac_left, tac_right, label, label_mask, label_all, dyn_data, dyn_mask, export_count, phase, ratio, save_path):
    if (len(tac_left) != len(label)) or (len(tac_left) != sum(ratio)):
        print ('check data length!!!')

    count = 0
    for p in range(len(phase)):
        count += ratio[p]
        for n in range(count-ratio[p], count):
            # remove incorrect insertion label
            tac_mask = np.zeros((label[n].shape[0], len(name_list), 9, 22)) # update number of tasks
            tac_mask[:, nt, :, :] = 1
            to_save = [contrain[0], contrain[1], tac_left[n], tac_right[n], tac_mask, label[n], label_mask[n]]

            if dyn_data == []:
                to_save.append(np.zeros((label[n].shape[0], 1)))
                to_save.append(np.zeros((label[n].shape[0], len(name_list)))) # update number of tasks
            else: 
                to_save.append(np.reshape(dyn_data[n], (-1, 1)))
                to_save.append(dyn_mask[n])

            path = save_path + phase[p] + '/' + str(export_count[p][-1]) + '.p'
            pickle.dump(to_save, open(path, "wb"))
            print ('dumped', save_path, to_save[6].shape, to_save[-3].shape, to_save[-2].shape, to_save[-1].shape)
            label_all[p] = np.concatenate((label_all[p], label[n]), axis=0)
            export_count[p].append(export_count[p][-1] + to_save[2].shape[0])
    
    return export_count, label_all






# task_list = ['_01', '_02', '_03', '_04', '_05', '_06']
# name_list = ['01_bolt', '02_screw', '03_case', '04_wrench', '05_pitcher_right', '06_pitcher_left']
# dyn_param_list = [-1, -1, 1, 1, 1, 1]
# label_count_list = [4, 5, 4, 6, 4, 0]

# task_list = ['_01', '_02', '_03', '_04', '_05']
# name_list = ['01_bolt', '02_screw', '03_case', '04_wrench', '05_pitcher_right']
# dyn_param_list = [-1, -1, 1, 1, 1]
# label_count_list = [4, 5, 4, 6, 4]

# task_list = ['_01', '_02', '_03',  '_05']
# name_list = ['01_bolt', '02_screw', '03_case', '05_pitcher_right']
# dyn_param_list = [-1, -1, 1, 1]
# label_count_list = [4, 5, 4, 4]


# task_list = ['_01', '_03', '_04', '_05']
# name_list = ['01_bolt', '03_case', '04_wrench', '05_pitcher_right']
# dyn_param_list = [-1, 1, 1, 1]
# label_count_list = [4, 4, 6, 4]

# task_list = ['_01', '_02', '_04', '_05']
# name_list = ['01_bolt', '02_screw', '04_wrench', '05_pitcher_right']
# dyn_param_list = [-1, -1, 1, 1]
# label_count_list = [4, 5, 6, 4]

# task_list = ['_01', '_02', '_03', '_04', ]
# name_list = ['01_bolt', '02_screw', '03_case', '04_wrench']
# dyn_param_list = [-1, -1, 1, 1]
# label_count_list = [4, 5, 4, 6]

# task_list = ['_05']
# name_list = ['05_pitcher_right']
# dyn_param_list = [1]
# label_count_list = [4]

# task_list = ['_02', '_03', '_04', '_05']
# name_list = [ '02_screw', '03_case', '04_wrench', '05_pitcher_right']
# dyn_param_list = [-1, 1, 1, 1]
# label_count_list = [5, 4, 6, 4]

# task_list = ['_01']
# name_list = ['01_bolt']
# dyn_param_list = [-1]
# label_count_list = [4]

# task_list = ['_02']
# name_list = ['02_screw']
# dyn_param_list = [-1]
# label_count_list = [5]

# task_list = ['_03']
# name_list = ['03_case']
# dyn_param_list = [1]
# label_count_list = [4]

# task_list = ['_04']
# name_list = ['04_wrench']
# dyn_param_list = [1]
# label_count_list = [6]

save_path = './dataset/all_dominant_05_p6/'

mode = 1 # free, dominant, non-dominant, 2-hand
viz = False
scale_bool = False

n_mode = 4 # number of data collection mode
phase = ['train', 'val', 'test'] # 7:2:1
export_count = [[0], [0], [0]]
n_labels = sum(label_count_list)
label_all = [np.zeros((1, n_labels)), np.zeros((1, n_labels)), np.zeros((1, n_labels))]


for nt, task in enumerate(task_list):
    name = name_list[nt]
    main_path = './data/processed/' + name + '/'
    print (name)

    # read log file
    df = pd.read_csv(main_path + 'log_p.csv', sep=',', header=0)
    log = df.to_numpy()
    n_file = log.shape[0]

    for f in range(n_file):
        # print (f)
        # extract meta data
        date = str("{:04d}".format(np.int16(log[f, 0])))
        p = str("{:02d}".format(np.int16(log[f, 1])))
        
        if np.int16(log[f, 1]) != 0: 
            p += task
            if log[f, 2] != 0:
                p += '_'
                p += str(np.int16(log[f, 2]))

            # load files
            if task == '_01' or task == '_02':
                tac_left_path = main_path + 'rec' + p + '_left.p'
                tac_right_path = main_path + 'rec' + p + '_right.p'
            else:
                pp = str("{:02d}".format(np.int16(log[f, 1]))) + '_dyn'
                if log[f, 2] != 0:
                    pp += '_'
                    pp += str(np.int16(log[f, 2]))
                tac_left_path = main_path + 'rec' + pp + '_left.p'
                tac_right_path = main_path + 'rec' + pp + '_right.p'

            tac_paths = [tac_left_path, tac_right_path]
            label_path = main_path + 'rec' + p + '_arr.mat'
            st_frame = np.int16(log[f, 3 + n_mode])
            tac_left, tac_right, label_data = load_files(tac_paths, label_path, st_frame)
            print (p)


            #load dynamic data
            if name == '03_case':
                dyn_data = pickle.load(open(main_path + 'rec' + pp + '_mocap.p', 'rb'))
                dyn_data = dyn_data[0][np.int16(log[f, 11])]
                sted = [0, 4] # set the start and end label
            if name == '04_wrench':
                dyn_data = pickle.load(open(main_path + 'rec' + pp + '_mocap.p', 'rb'))
                dyn_data = dyn_data[3][np.int16(log[f, 11])]
                sted = [1, 5] # set the start and end label
            elif name == '05_pitcher_right':
                # dyn_data = pickle.load(open(main_path + 'rec' + pp + '_mocap.p', 'rb'))
                # dyn_data = dyn_data[4][np.int16(log[f, 11])]
                sted = [1, 4] # set the start and end label
                dyn_data = pickle.load(open(main_path + 'rec' + pp + '_scale.p', 'rb'))
            elif name == '06_pitcher_left':
                dyn_data = pickle.load(open(main_path + 'rec' + pp + '_mocap.p', 'rb'))
                dyn_data = dyn_data[5][np.int16(log[f, 11])]
                sted = [1, 4] # set the start and end label
                # dyn_data = pickle.load(open(main_path + 'rec' + pp + '_scale.p', 'rb')) # scale data

            # print ('dyn data loaded', dyn_data.shape)

            mode_count = []
            for nn in range(n_mode):
                mode_count.append(np.int16(log[f, 3+nn])) # free, dominant, non-dominant, 2-hand

            # split by mode
            tac_left_by_mode, tac_right_by_mode, label_by_mode = split_by_mode(tac_left, tac_right, label_data, mode_count)
            tac_left = tac_left_by_mode[mode]
            tac_right = tac_right_by_mode[mode]

            # split dynamic parameters by mode
            if dyn_param_list[nt] > 0:
                dyn_by_mode, dyn_mask_by_mode = split_dyn_by_mode(tac_left, tac_right, label_data, sted, dyn_data, name_list, nt, mode_count)
                dyn = dyn_by_mode[mode]
                dyn_mask = dyn_mask_by_mode[mode]  
            else:
                dyn = []
                dyn_mask = []

            # convert time stamp label to stage label, n_stage = n_label-1
            label, label_mask = split_by_stage(label_by_mode[mode], label_count_list, nt, name_list)

            if viz:
                # plt.imshow(label[0], aspect=0.01)
                # plt.show()
                # plt.imshow(label_mask[0], aspect=0.01)
                # plt.show()
                if dyn_mask != []:
                    # plt.imshow(dyn[0], aspect=0.01)
                    # plt.show()
                    plt.imshow(dyn_mask[0], aspect=0.01)
                    plt.show()

                    plt.plot(dyn[1])
                    plt.show()

            # dump as train/val/test set
            ratio = [np.int16(log[f, 4 + n_mode]), np.int16(log[f, 5 + n_mode]), np.int16(log[f, 6 + n_mode])]

            constrain_list = os.listdir(main_path + 'constrain/')
            idx = random.randint(0, len(constrain_list)-1)
            contrain = pickle.load(open(main_path + 'constrain/' + constrain_list[idx], 'rb'))

            export_count, label_all = export(name_list, nt, contrain, tac_left, tac_right, label, label_mask, label_all, dyn, dyn_mask, export_count, phase, ratio, save_path)
            # figure out data balancing
            # export format [constrain_left (200), constrain_right, tac_left, tac_right, tac_mask, label, label_mask, dyn, dyn_mask]

            # print (name, export_count)

        else:
            # dump log for each phase
            for p in range(len(phase)):
                if export_count[p][-1] != 0:
                    pickle.dump(export_count[p], open(save_path + phase[p] + '/log.p', "wb"))
                    print (phase[p], export_count[p])

            # figure out data balancing      
            for i in range(len(label_all)):
                if export_count[i][-1] != 0:
                    cls_weight = np.sum(label_all[i]) / (np.sum(label_all[i], axis=0) + 1e-4)
                    print (cls_weight)
                    sample_weight = []
                    for s in range(1, label_all[i].shape[0]):
                        ind = np.argmax(label_all[i][s, :])
                        sample_weight.append(cls_weight[ind])
                    pickle.dump(sample_weight, open(save_path + phase[i] + '/sample_weight.p', "wb"))

# dump log for each phase
for p in range(len(phase)):
    if export_count[p][-1] != 0:
        pickle.dump(export_count[p], open(save_path + phase[p] + '/log.p', "wb"))
        print (phase[p], export_count[p])

# figure out data balancing      
for i in range(len(label_all)):
    if export_count[i][-1] != 0:
        cls_weight = np.sum(label_all[i]) / (np.sum(label_all[i], axis=0) + 1e-4)
        print (cls_weight)
        sample_weight = []
        for s in range(1, label_all[i].shape[0]):
            ind = np.argmax(label_all[i][s, :])
            sample_weight.append(cls_weight[ind])
        pickle.dump(sample_weight, open(save_path + phase[i] + '/sample_weight.p', "wb"))

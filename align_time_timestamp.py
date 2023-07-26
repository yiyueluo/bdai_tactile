import cv2
import numpy as np
import pandas as pd
from utils import *
import matplotlib as mpl
import h5py


''' main '''

mpl.use('Agg')

main_path = './data/'
save_path = './data/'

viz_bool = False
shift_bool = True
normalize_bool = True

mocap_bool = True
scale_bool = True

# load tactile map layout
tac_layout_left_path = './data/common/hand_layout_left.csv'
tac_layout_right_path = './data/common/hand_layout_right.csv'
df = pd.read_csv(tac_layout_left_path, sep=',', header=0)
tac_layout_left = df.to_numpy() #[index 0, index 1, element no.]
df = pd.read_csv(tac_layout_right_path, sep=',', header=0)
tac_layout_right = df.to_numpy() #[index 0, index 1, element no.]
tac_layout = [tac_layout_left, tac_layout_right]

# read log
df = pd.read_csv(main_path + 'log.csv' , sep=',', header=0)

for n, filename in enumerate(df['name']):

    # extract start time stamps
    tac_st_time = convert_time(df['tactile'][n]) # unit: sec
    cam0_st_time = convert_time(df['cam0'][n]) #+ 10 # offset
    cam1_st_time = convert_time(df['cam1'][n])  #+ 4 #+ 5.5

    if df['mocap'][n] != '0':
        mocap_bool = True
    if df['scale'][n] != 0:
        scale_bool = True

    print (mocap_bool, scale_bool)
    # read videos
    videos_path = [main_path + filename + '_cam0.mp4', main_path + filename + '_cam1.mp4']
    cam0_fps, cam0_n_frames = read_video(videos_path[0])
    cam1_fps, cam1_n_frames = read_video(videos_path[1])

    cam0_ts = np.asanyarray(np.arange(0, 1/cam0_fps * cam0_n_frames, 1/cam0_fps)) # relative timestamps, unit: s
    cam1_ts = np.asanyarray(np.arange(0, 1/cam1_fps * cam1_n_frames, 1/cam1_fps))
    cam0_ts += cam0_st_time # system timestamps, unit: s
    cam1_ts += cam1_st_time
  
    # read tactile data 
    tac_fps, tac_ts, tac_data = read_tactile_csv_with_timestamp(main_path + filename + '_tac.csv') # relative timestamps, unit: s
    tac_ts += tac_st_time

    # normalize tactile data
    print ("min:", np.amin(tac_data), 'max:', np.amax(tac_data), 'mean:', np.mean(tac_data))
    lo = np.amin(tac_data)
    # lo = -1
    hi = 40
    if normalize_bool:
        tac_data = (tac_data - lo) / (hi - lo)
        tac_data = np.where(tac_data<0, 0, tac_data)
    print ('tactile normalized.', "min:", np.amin(tac_data), 'max:', np.amax(tac_data))

    # read mocap data
    if mocap_bool:
        mocap_st_time = convert_time(df['mocap'][n]) # + 3 #+ 9
        # case, hand_right, hand_left, wrench, pitch_left, pitch_right
        mocap_fps, mocap_ts, mocap_data = read_mocap(main_path + filename + '_mocap.csv', n_body=6) # mocap data [[x1, y1, z1], [x2, y2, z2]...]
        mocap_ts += mocap_st_time

        lo = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [100, 100, 100], [0, 0, 0], [0, 0, 0]]
        hi = [[0.8, 0.8, 0.8], [1, 1, 1], [1, 1, 1], [700, 700, 700], [500, 500, 500], [500, 500, 500]]
        if normalize_bool:
            for i in range(len(mocap_data)):
                for j in range(len(mocap_data[0])):
                    # print ('mocap.', "min:", np.amin(mocap_data[i][j]), 'max:', np.amax(mocap_data[i][j]))
                    mocap_data[i][j] = (mocap_data[i][j] - lo[i][j]) / (hi[i][j] - lo[i][j])
                    mocap_data[i][j] = np.where(mocap_data[i][j]<0, 0, mocap_data[i][j])
                    print ('mocap normalized.', "min:", np.amin(mocap_data[i][j]), 'max:', np.amax(mocap_data[i][j]))

    # read and normalize scale data
    if scale_bool:
        scale_ts, scale_data = read_scale(main_path + filename + '_scale.hdf5')
        print (np.amin(scale_data), np.amax(scale_data))
        # lo = -1
        # hi = np.amax(scale_data)
        lo = 1e6
        hi = 1.5e6
        if normalize_bool:
            scale_data = (scale_data - lo) / (hi - lo)
            scale_data = np.where(scale_data<0, 0, scale_data)
        print ('scale normalized.', "min:", np.amin(scale_data), 'max:', np.amax(scale_data))

    # print ('start time', tac_ts[0], cam0_ts[0], cam1_ts[0], mocap_ts[0])
    # print ('end time', tac_ts[-1], cam0_ts[-1], cam1_ts[-1], mocap_ts[-1]) 

    # extract time difference
    if shift_bool:
        if not (df['cam0_clap_frame'][n] == 0 and df['tac_clap_frame'][n] == 0 and df['cam1_clap_frame'][n] == 0):
            cam0_time_shift = cam0_ts[df['cam0_clap_frame'][n]] - tac_ts[df['tac_clap_frame'][n]] #regarding to cam1 because cam1 align with mocap
            cam0_ts -= cam0_time_shift     
            cam1_time_shift = cam1_ts[df['cam1_clap_frame'][n]] - tac_ts[df['tac_clap_frame'][n]] #regarding to cam1 because cam1 align with mocap
            cam1_ts -= cam1_time_shift   

    if mocap_bool:
        if not (df['mocap_frame'][n] == 0 and df['cam1_mocap_frame'][n] == 0):
            mocap_time_shift = mocap_ts[df['mocap_frame'][n]] - cam1_ts[df['cam1_mocap_frame'][n]] #regarding to cam1 because cam1 align with mocap
            mocap_ts -= mocap_time_shift    
    if scale_bool:
        if not (df['scale_frame'][n] == 0 and df['cam1_scale_frame'][n] == 0):
            scale_time_shift = scale_ts[df['scale_frame'][n]] - cam1_ts[df['cam1_scale_frame'][n]] #regarding to cam1 because cam1 align with mocap
            scale_ts -= scale_time_shift 
    print (cam0_time_shift, cam1_time_shift, mocap_time_shift, scale_time_shift)


    # align tactile data with other assets
    to_align_ts = [cam0_ts, cam1_ts]
    if mocap_bool:
        to_align_ts.append(mocap_ts)
    if scale_bool:
        to_align_ts.append(scale_ts)

    align_ts_index = align_ts_with_timestamp(tac_ts, to_align_ts)

    # export index data
    pickle.dump(align_ts_index, open(save_path + filename + '_aligned_index.p', "wb"))
    print ('aligned index exported')

    # export tactile data
    export_tac(tac_data, tac_layout_left, save_path + filename + '_left.p')
    export_tac(tac_data, tac_layout_right, save_path + filename + '_right.p')
    print ('aligned tactile exported')

    # export mocap data
    if mocap_bool:
        mocap_aligned_index = align_ts_index[2]
        # print (mocap_aligned_index)
        export_mocap(mocap_data, mocap_aligned_index, save_path + filename + '_mocap.p')
        print ('aligned mocap exported')

    # export scale data
    if scale_bool:
        scale_aligned_index = align_ts_index[3]
        export_scale(scale_data, scale_aligned_index, save_path + filename + '_scale.p')
        print ('aligned scale exported')


    # generate viz

    # path = save_path + filename + '_aligned.avi' 
    # videos_aligned_index = [align_ts_index[0], align_ts_index[1]]
    # if not (mocap_bool or scale_bool):
    #     viz_static_with_timestamp(tac_data, videos_path, videos_aligned_index, tac_layout, path, viz_bool)

    # else:
    #     aligned_dynamic_data = []
    #     aligned_dynamic_label = []
    #     aligned_dynamic_index = []
    #     if mocap_bool:
    #         aligned_dynamic_index.append(mocap_aligned_index)
    #         # for i in range(3):
    #         aligned_dynamic_data.append(mocap_data[0][0][mocap_aligned_index]) # case z
    #         aligned_dynamic_label.append('case z')
    #         aligned_dynamic_data.append(mocap_data[3][2][mocap_aligned_index]) # wrench x
    #         aligned_dynamic_label.append('wrench x')
    #         aligned_dynamic_data.append(mocap_data[4][2][mocap_aligned_index]) # pitcher_left x
    #         aligned_dynamic_label.append('pitcher_right x')
    #         aligned_dynamic_data.append(mocap_data[5][2][mocap_aligned_index]) # pitcher_right x
    #         aligned_dynamic_label.append('pitcher_left x')


    #     if scale_bool:
    #         aligned_dynamic_index.append(scale_aligned_index)
    #         aligned_dynamic_data.append(scale_data[scale_aligned_index])
    #         aligned_dynamic_label.append('scale')
    #     viz_dynamic_with_timestamp(tac_data, videos_path, videos_aligned_index, aligned_dynamic_index, tac_layout, aligned_dynamic_data, aligned_dynamic_label, path, viz_bool)


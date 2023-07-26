import cv2
import numpy as np
import pandas as pd
from utils import *
import matplotlib as mpl

# reload(matplotlib)
# matplotlib.use('Agg')
mpl.use('Agg')

date = '0707'
n_participant = '06'
n_task = '02'
comment = '_2' #_1
viz_bool = False

# markers
tac_marker_frame = 68  #array index
videos_marker_frame = [9537,	9323]  #array index

# read videos
main_path = './data/'
videos_path = [main_path + date + '_rec' + n_participant + '_' + n_task + '_cam0.mp4', main_path + date + '_rec' + n_participant + '_' + n_task + '_cam1.mp4']

cam0_fps, cam0_n_frames = read_video(videos_path[0])
cam1_fps, cam1_n_frames = read_video(videos_path[1])

cam0_ts = np.arange(0, 1/cam0_fps * cam0_n_frames, 1/cam0_fps)
cam1_ts = np.arange(0, 1/cam1_fps * cam1_n_frames, 1/cam1_fps)
videos_ts = [cam0_ts, cam1_ts]

# read tactile data 
tac_fps, tac_ts, tac_data = read_tactile_csv(main_path + date + '_rec' + n_participant + '_' + n_task + comment + '.csv')

# normalize tactile data
print ("min:", np.amin(tac_data), 'max:', np.amax(tac_data), 'mean:', np.mean(tac_data))
# lo = np.amin(tac_data)
lo = -1
hi = 40
if not viz_bool:
    tac_data = (tac_data - lo) / (hi - lo)
    tac_data = np.where(tac_data<0, 0, tac_data)
print ('normalized.', "min:", np.amin(tac_data), 'max:', np.amax(tac_data))


# align tactile data with videos
videos_aligned_ts, videos_aligned_index = align_ts(tac_marker_frame, videos_marker_frame, tac_ts, videos_ts)


# load tactile map layout
tac_layout_left_path = main_path + 'common/hand_layout_left.csv'
tac_layout_right_path = main_path + 'common/hand_layout_right.csv'

df = pd.read_csv(tac_layout_left_path, sep=',', header=0)
tac_layout_left = df.to_numpy() #[index 0, index 1, element no.]

df = pd.read_csv(tac_layout_right_path, sep=',', header=0)
tac_layout_right = df.to_numpy() #[index 0, index 1, element no.]

tac_layout = [tac_layout_left, tac_layout_right]


#export data
save_path = main_path + 'processed/rec' + n_participant + '_' + n_task + comment + '_left.p'
export_tac(tac_data, tac_layout_left, save_path)
save_path = main_path + 'processed/rec' + n_participant + '_' + n_task + comment + '_right.p'
export_tac(tac_data, tac_layout_right, save_path)


#generate viz

# save_path = main_path + 'processed/rec' + n_participant + '_' + n_task + comment + '_aligned.avi' 
# viz(tac_data, videos_path, videos_aligned_index, tac_layout, tac_marker_frame, save_path, viz_bool)


import cv2
import numpy as np
import pandas as pd
from utils import *

# read videos
main_path = './data/'
videos_path = [main_path + '0616_rec02_cam0.mp4', main_path + '0616_rec02_cam1.mp4']

cam0_fps, cam0_n_frames = read_video(videos_path[0])
cam1_fps, cam1_n_frames = read_video(videos_path[1])

cam0_ts = np.arange(0, 1/cam0_fps * cam0_n_frames, 1/cam0_fps)
cam1_ts = np.arange(0, 1/cam1_fps * cam1_n_frames, 1/cam1_fps)
videos_ts = [cam0_ts, cam1_ts]

# read tactile data 
tac_fps, tac_ts, tac_data = read_tactile_csv(main_path + '0616_rec02.csv')


# markers
tac_marker_frame = 69  #array index
videos_marker_frame = [605, 980]  #array index


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


#generate viz
save_path = main_path + '0616_rec02_aligned3.avi'
viz(tac_data, videos_path, videos_aligned_index, tac_layout, save_path, viz=False)


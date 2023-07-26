import cv2
import numpy as np
import pandas as pd

path = './data/0725_rec01_dyn_cam1.mp4'
video = cv2.VideoCapture(path)
fps = video.get(cv2.CAP_PROP_FPS)
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print (fps, n_frames)


for frame_no in range(1600, 24000):
    video.set(1,frame_no)
    res, curr_frame = video.read()
    curr_frame = cv2.resize(curr_frame, (720, 400))
    cv2.imshow('image', curr_frame)
    print (frame_no)
    cv2.waitKey(1)




# # while(video.isOpened()):
# #     frame_exists, curr_frame = video.read()
# #     if frame_exists:
# #         video_ts.append(video.get(cv2.CAP_PROP_POS_MSEC))
# #     else:
# #         break

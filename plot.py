import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from scipy.special import softmax
import time
import h5py
import math
import matplotlib as mpl
import scipy.io

c = ["#3951a2", "#5c91c2", "#fdb96b", '#f67948', "#e5756b", '#eea59f']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


'''plot ablation study on window size'''
# path = '../BDAI_ICRA/result.csv'
# df = pd.read_csv(path, sep=',', header=0)
# lt = 'ave-cls'
# mt = 'sd-cls'

# labels = ['ws = 1', 'ws = 20', 'ws = 40', 'ws = 60', 'ws = 80']
# l1 = [df[lt][5]]
# l2 = [df[lt][6]]
# l3 = [df[lt][7]]
# l4 = [df[lt][8]]
# l5 = [df[lt][9]]

# m1 = [df[mt][5]]
# m2 = [df[mt][6]]
# m3 = [df[mt][7]]
# m4 = [df[mt][8]]
# m5 = [df[mt][9]]


# barWidth = 0.25
# fig, ax = plt.subplots(figsize=(5, 5))
# # ax = fig.add_subplot(111)
# # ax.set_ylim((0,1.1))
# # ax.set_xlim((-0.5, 2.5))
# ax.tick_params(axis="y",direction="in")
# # ax.tick_params(axis="x",direction="in")

# # Set position of bar on X axis
# br1 = np.arange(len(l1))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
# br4 = [x + barWidth for x in br3]
# br5 = [x + barWidth for x in br4]

# # Make the plot
# ax.bar(br1, l1, color=c[0], width=barWidth, label=labels[0])
# ax.bar(br2, l2, color=c[1], width=barWidth, label=labels[1])
# ax.bar(br3, l3, color=c[2], width=barWidth, label=labels[2])
# ax.bar(br4, l4, color=c[4], width=barWidth, label=labels[3])
# ax.bar(br5, l5, color=c[5], width=barWidth, label=labels[4])

# ax.errorbar(br1, l1, m1, fmt="o", color=c[3])
# ax.errorbar(br2, l2, m2, fmt="o", color=c[3])
# ax.errorbar(br3, l3, m3, fmt="o", color=c[3])
# ax.errorbar(br4, l4, m4, fmt="o", color=c[3])
# ax.errorbar(br5, l5, m5, fmt="o", color=c[3])

# plt.legend()
# plt.show()



''' plot ablation study result 2''' 
# path = '../BDAI_ICRA/result.csv'
# df = pd.read_csv(path, sep=',', header=0)
# lt = 'ave-tac'
# mt = 'sd-tac'

# labels = ['All', 'Unseen participant', 'Unseen participant w/ single output modality', 'Unseen participant w/ single task']
# l1 = [df[lt][0]]
# l2 = [df[lt][4]]
# l3 = [df[lt][11]]
# l4 = [df[lt][10]]

# m1 = [df[mt][0]]
# m2 = [df[mt][4]]
# m3 = [df[mt][11]]
# m4 = [df[mt][10]]


# barWidth = 0.25
# fig, ax = plt.subplots(figsize=(5, 5))
# # ax = fig.add_subplot(111)
# # ax.set_ylim((0,1.1))
# # ax.set_xlim((-0.5, 2.5))
# ax.tick_params(axis="y",direction="in")
# # ax.tick_params(axis="x",direction="in")

# # Set position of bar on X axis
# br1 = np.arange(len(l1))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
# br4 = [x + barWidth for x in br3]

# # Make the plot
# ax.bar(br1, l1, color=c[0], width=barWidth, label=labels[0])
# ax.bar(br2, l2, color=c[1], width=barWidth, label=labels[1])
# ax.bar(br3, l3, color=c[2], width=barWidth, label=labels[2])
# ax.bar(br4, l4, color=c[4], width=barWidth, label=labels[3])

# ax.errorbar(br1, l1, m1, fmt="o", color=c[3])
# ax.errorbar(br2, l2, m2, fmt="o", color=c[3])
# ax.errorbar(br3, l3, m3, fmt="o", color=c[3])
# ax.errorbar(br4, l4, m4, fmt="o", color=c[3])

# plt.legend()
# plt.show()


# ''' plot ablation study result 1''' 
# path = '../BDAI_ICRA/result.csv'
# df = pd.read_csv(path, sep=',', header=0)
# lt = 'ave-obj'
# mt = 'sd-obj'

# labels = ['All', 'Single output category', 'Single task', 'Unseen task (w/ refined decoder)']
# l1 = [df[lt][0]]
# l2 = [df[lt][1]]
# l3 = [df[lt][2]]
# l4 = [df[lt][3]]

# m1 = [df[mt][0]]
# m2 = [df[mt][1]]
# m3 = [df[mt][2]]
# m4 = [df[mt][4]]


# barWidth = 0.25
# fig, ax = plt.subplots(figsize=(5, 5))
# # ax = fig.add_subplot(111)
# # ax.set_ylim((0,1.1))
# # ax.set_xlim((-0.5, 2.5))
# ax.tick_params(axis="y",direction="in")
# # ax.tick_params(axis="x",direction="in")

# # Set position of bar on X axis
# br1 = np.arange(len(l1))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
# br4 = [x + barWidth for x in br3]

# # Make the plot
# ax.bar(br1, l1, color=c[0], width=barWidth, label=labels[0])
# ax.bar(br2, l2, color=c[1], width=barWidth, label=labels[1])
# ax.bar(br3, l3, color=c[2], width=barWidth, label=labels[2])
# ax.bar(br4, l4, color=c[4], width=barWidth, label=labels[3])

# ax.errorbar(br1, l1, m1, fmt="o", color=c[3])
# ax.errorbar(br2, l2, m2, fmt="o", color=c[3])
# ax.errorbar(br3, l3, m3, fmt="o", color=c[3])
# ax.errorbar(br4, l4, m4, fmt="o", color=c[3])

# plt.legend()
# plt.show()


'''plot ablation on masked area'''
# m = np.empty((9, 22))
# m[:2, 11:13] = np.nan
# m[7:, 11:13] = np.nan
# m[0, 16:] = np.nan

# m[2:4, 11:13] = 0.832
# m[4:7, 11:13] = 0.841
# m[:3, 13:16] = 0.784
# m[3:4, 13:16] = 0.839
# m[4:5, 13:16] = 0.856
# m[1:3, 16:18] = 0.785 
# m[3:4, 16:18] = 0.838 
# m[4:5, 16:18] = 0.826 
# m[1:3, 18:20] = 0.816 
# m[3:4, 18:20] = 0.848 
# m[4:5, 18:20] = 0.844 
# m[1:3, 20:22] = 0.83 
# m[3:4, 20:22] = 0.847 
# m[4:5, 20:22] = 0.845 
# m[5:6, 13:] = 0.815 
# m[6:, 13:18] = 0.853 
# m[6:, 18:22] = 0.839 

# m = m[:, 11:]
# m = 1 - (m - 0.784)/(0.89 - 0.784)
# print (np.amin(m), np.amax(m))
# plt.imshow(m, vmin=0, vmax=1, cmap='viridis')
# plt.axis('off')
# plt.colorbar()
# plt.show()




# ''' plot subsample '''

subsample_list = [1, 2, 3, 4]
acc_list = [0.892, 0.781, 0.746, 0.404]
acc_sd_list = []

dyn_list = []
dyn_sd_list = []

plt.plot(subsample_list, acc_list, marker='o')
plt.ylim(0, 1)
plt.show()


''' plot window size '''

# window_list = [1, 20, 40, 60, 80, 100]
# acc_list = [0.8,  0.82,  0.892, 0.845, 0.84, 0.82]

# plt.plot(window_list, acc_list, marker='o')
# plt.ylim(0.6, 1)
# plt.show()




''' extract tactile examples '''
# # main_path = './data/processed/01_bolt/'
# main_path = './data/'
# name = '0724_rec01_dyn'

# tac_left = pickle.load(open(main_path + name + '_left.p', 'rb'))
# tac_right = pickle.load(open(main_path + name + '_right.p', 'rb'))

# for i in range(3846, 15000):
#     print (i)
#     plt.imshow(tac_left[i, :, :], vmin=0, vmax=0.5, cmap='binary')
#     plt.axis('off')
#     plt.show()
#     plt.imshow(tac_right[i, :, :], vmin=0, vmax=0.5, cmap='binary')
#     plt.axis('off')
#     plt.show()




''' plot video tape'''
# video_path = './data/processed/01_bolt/'
# video_clips = ['rec01_01', 'rec04_01', 'rec07_01']
# rd = [8, 5, 5]
# st_frame = [267, 513, 194]
# count_list = [0]
# st_list = []

# count = 0
# # for v in range(1):
# for v in range(len(video_clips)):
#     label_path = video_path + video_clips[v] +  '_arr.mat'
#     mat = scipy.io.loadmat(label_path)
#     label_data = mat['LabelData']
#     extracted_st = (np.where(label_data[:, 0]==1))[0][rd[v]] 
#     extracted_ed = (np.where(label_data[:, -1]==1))[0][rd[v]]

#     count = count + (extracted_ed - extracted_st)
#     count_list.append(count)
#     st_list.append(extracted_st)

# count_list = np.asanyarray(count_list)
# print (count_list)


# plt.ioff()
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# out = cv2.VideoWriter(save_path + name + '_2.avi', fourcc, 30, (1400,800))
# print ('Video streaming')

# for i in range(3000, tactile.shape[0]):

#     video_ind = np.where(count_list<=i)[0][-1] 
#     # print (video_ind)

#     cam = cv2.VideoCapture(video_path + video_clips[video_ind] + '_aligned.avi')
#     cam.set(1, st_list[video_ind] + i - count_list[video_ind])
#     res, curr_frame = cam.read()
#     curr_frame = curr_frame[:200, 350:, :3]
#     curr_frame = cv2.resize(curr_frame, (700, 400))

#     if np.sum(gt[i, :]) != 0:

#         # plot probabilities
#         plot_labels = ['Ground truth', 'Prediction']
#         plot_labels2 = ['picking up bolt', 'inserting bolt', 'threading', 'releasing' ]

#         # pred[i, :] = (pred[i, :] - np.min(pred[i, :])) / (np.max(pred[i, :]) - np.min(pred[i, :]))
#         pred[i, :] = softmax(pred[i, :])

#         # l = [[], []] #[gt, pred]
#         # for j in range(gt.shape[1]):
#         #     l[0].append(gt[i, j])
#         #     l[1].append(pred[i, j])

#         l = [[]] #[gt, pred]
#         for j in range(gt.shape[1]):
#             l[0].append(pred[i, j])


#         barWidth = 0.25
#         fig, ax = plt.subplots(figsize=(7, 8))
#         # ax = fig.add_subplot(111)
#         ax.set_ylim((0,1.1))
#         # ax.set_xlim((-0.5, 2.5))
#         ax.tick_params(axis="y",direction="in")
#         ax.tick_params(axis="x",direction="in")

#         # br = [[], []]
#         # br[0] = np.arange(len(l[0]))
#         # br[1] = barWidth + br[0]

#         br = [[]]
#         br[0] = np.arange(len(l[0]))


#         for j in range(len(l)):
#             ax.bar(br[j], l[j], width=barWidth, label=plot_labels[j])

#         plt.xticks(range(gt.shape[1]), plot_labels2)
#         id_tick_change_colour = np.argmax(gt[i, :])

#         plt.setp(ax.get_xticklabels()[id_tick_change_colour], color='red')

#         plt.xlabel('Class')
#         plt.ylabel('Probability')
#         # plt.xticks(rotation=45)
#         # plt.legend()
#         plt.title('Classification probability')
#         fig.canvas.draw()
#         # plt.show()
#         frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#         prob = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 1000 x 1000


#         # plot tactile
#         tactile = np.where(tactile==0, np.nan, tactile)

#         fig = plt.figure(figsize=(4, 4))
#         plt.imshow(tactile[i, 0, :, :11], vmin=0, vmax=0.5, cmap='viridis')
#         plt.title('Left hand tactile frames')
#         plt.axis('off')
#         fig.canvas.draw()
#         # plt.show()
#         frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#         tac_plot_left = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
#         tac_plot_left = cv2.resize(tac_plot_left, (350, 400))

#         fig = plt.figure(figsize=(4, 4))
#         plt.imshow(tactile[i, 0, :, 11:], vmin=0, vmax=0.5, cmap='viridis')
#         plt.title('Right hand tactile frames')
#         plt.axis('off')
#         fig.canvas.draw()
#         # plt.show()
#         frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#         tac_plot_right = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
#         tac_plot_right = cv2.resize(tac_plot_right, (350, 400))

#         tac = np.concatenate((tac_plot_left, tac_plot_right), axis=1) 
#         # print (tac.shape)
#         img = np.concatenate((curr_frame, tac), axis=0)
#         # print (curr_frame.shape)
#         img = np.concatenate((img, prob), axis=1)
#         # print (prob.shape)
#         # # img = np.concatenate((tac, prob), axis=0)

#         # img = video_arr[i, :, :, :]
#         # print (img.shape)
#         out.write(img)
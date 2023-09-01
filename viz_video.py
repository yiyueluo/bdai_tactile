import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io
from scipy.special import softmax

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


main_path = './dataset/all_dominant/predictions/'
name = 'eval_exp22'
data = pickle.load(open(main_path + name + '.p', "rb"))
# data format [tac_input_list, label_list, label_mask_list, cls_pred_list, tac_output_list, tac_pred_list, tac_mask_list, dyn_list, dyn_pred_list, dyn_mask_list, feature_list]

for i in range(11):
    print (data[i].shape)

tac_input = data[0]
label_gt = data[1]
label_mask = data[2]
label_pred = data[3]
tac_gt = data[4]
tac_pred = data[5]
tac_mask = data[6]
dyn_gt = data[7]
dyn_pred = data[8]
dyn_mask = data[9]
feature = data[10]

plot_cls = False
plot_dyn = False
plot_tac = False


### classification ####

label_count = [4, 5, 4, 6, 4]

cls_gt = label_gt[1:, -1, :] * label_mask[1:, -1, :]
cls_pred = label_pred[1:, -1, :] * label_mask[1:, -1, :]
# print (cls_gt.shape, cls_pred.shape)

# fig, ax = plt.subplots(1,2) 
# ax[0].imshow(cls_gt,  aspect=0.01)
# ax[1].imshow(cls_pred,  aspect=0.01)
# plt.show()

tactile = tac_input
main_path = './data/processed/'
save_path = './viz/'


# video_path = '01_bolt/'
# video_clips = ['rec01', 'rec07']
# comments = ['', '']
# rd = [9, 6]
# st_frame = [267, 194] 
# comp = 0 # previous count sum
# seq_n = '_01' 
# cls_pred = cls_pred[:, :4] # define task specific label
# cls_gt = cls_gt[:, :4] # define task specific label
# plot_labels2 = ['picking up', 'inserting', 'threading', 'releasing' ]
# plot_tac = True


video_path = '02_screw/'
video_clips = ['rec01', 'rec09']
comments = ['', '']
rd = [6, 5]
st_frame = [263, 571] 
comp = 1653 # previous count sum
seq_n = '_02' 
cls_pred = cls_pred[:, 4:9] # define task specific label
cls_gt = cls_gt[:, 4:9] # define task specific label
plot_labels2 = ['picking up', 'inserting bolt', 'threading', 'releasing', 'putting down']
plot_tac = True


# video_path = '03_case/'
# video_clips = ['rec01', 'rec08']
# comments = ['', '']
# rd = [8, 3]
# st_frame = [0, 0] 
# comp = 3164 # previous count sum
# seq_n = '_03' 
# cls_pred = cls_pred[:, 9:13] # define task specific label
# cls_gt = cls_gt[:, 9:13] # define task specific label
# plot_labels2 = ['unlocking right', 'unlocking left', 'lifting', 'releasing' ]

# video_path = '03_case/'
# video_clips = ['rec12', 'rec13']
# comments = ['', '']
# rd = [6, 8]
# st_frame = [0, 0] 
# comp = 3773 # previous count sum
# seq_n = '_03' 
# cls_pred = cls_pred[:, 9:13] # define task specific label
# cls_gt = cls_gt[:, 9:13] # define task specific label
# plot_labels2 = ['unlocking right', 'unlocking left', 'lifting', 'releasing' ]
# dyn_index = 2
# dyn_range = 100
# plot_dyn = True

# video_path = '04_wrench/'
# video_clips = ['rec01', 'rec07', 'rec13']
# comments = ['', '', '']
# rd = [7, 8, 4]
# st_frame = [0, 0, 0] 
# comp = 3544 # previous count sum
# seq_n = '_04' 
# cls_pred = cls_pred[:, 13:19] # define task specific label
# cls_gt = cls_gt[:, 13:19] # define task specific label
# plot_labels2 = ['picking up', 'rotating left', 'straightening from left', 'rotating right', 'straightening from left', 'releasing']

# video_path = '04_wrench/'
# video_clips = ['rec12', 'rec13']
# comments = ['_2', '']
# rd = [3, 5]
# st_frame = [0, 0] 
# comp = 4277 # previous count sum
# seq_n = '_04' 
# cls_pred = cls_pred[:, 13:19] # define task specific label
# cls_gt = cls_gt[:, 13:19] # define task specific label
# plot_labels2 = ['picking up', 'rotating left', 'straightening from left', 'rotating right', 'straightening from left', 'releasing']
# dyn_index = 3
# dyn_range = 180
# plot_dyn = True



# video_path = '05_pitcher_right/'
# video_clips = ['rec01', 'rec01', 'rec06']
# comments = ['', '_2', '_2']
# rd = [6, 8, 7]
# st_frame = [0, 0, 0] 
# comp = 4533 # previous count sum
# seq_n = '_05' 
# cls_pred = cls_pred[:, 19:] # define task specific label
# cls_gt = cls_gt[:, 19:] # define task specific label
# plot_labels2 = ['picking up', 'pouring', 'straightening', 'putting down']


# video_path = '05_pitcher_right/'
# video_clips = ['rec12', 'rec05']
# comments = ['_2', '_2']
# rd = [6, 6]
# st_frame = [0, 14700] 
# comp = 5376 # previous count sum
# seq_n = '_05' 
# cls_pred = cls_pred[:, 19:] # define task specific label
# cls_gt = cls_gt[:, 19:] # define task specific label
# plot_labels2 = ['picking up', 'pouring', 'straightening', 'putting down']
# dyn_index = 4
# dyn_range = 1400
# plot_dyn = True


count_list = [0]
st_list = []

count = 0

# for v in range(1):
for v in range(len(video_clips)):
    label_path = main_path + video_path + video_clips[v] +  seq_n + comments[v] + '_arr.mat'
    mat = scipy.io.loadmat(label_path)
    label_data = mat['LabelData']
    extracted_st = (np.where(label_data[:, 0]==1))[0][rd[v]-1] 
    extracted_ed = (np.where(label_data[:, -1]==1))[0][rd[v]-1]

    count = count + (extracted_ed - extracted_st)
    count_list.append(count)
    st_list.append(extracted_st)

count_list = np.asanyarray(count_list) + comp
print (count_list)

plt.ioff()
mpl.use('Agg')


### plot cls #####
if plot_cls:
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_path + name + seq_n + '_cls.avi', fourcc, 30, (1400,800))
    print ('Video streaming')


    for i in range(count_list[0], count_list[-1]+1):
        if i%50 == 0:
            print (i)

        video_ind = np.where(count_list<=i)[0][-1] 
        # print (video_ind)

        if video_path == '03_case/' or video_path == '04_wrench/' or video_path == '05_pitcher_right/':
            # print ('here')
            cam = cv2.VideoCapture('./data/dynamic_examples_raw_data/' + video_clips[video_ind] + '_dyn' + comments[video_ind] + '_aligned.avi')
        else:
            cam = cv2.VideoCapture(main_path + video_path + video_clips[video_ind] + seq_n + comments[v] + '_aligned.avi')
        
        cam.set(1, st_list[video_ind] + i - count_list[video_ind])
        res, curr_frame = cam.read()
        curr_frame = curr_frame[:200, 350:, :3]
        curr_frame = cv2.resize(curr_frame, (700, 400))


        if np.sum(cls_gt[i, :]) != 0:

            # plot probabilities
            plot_labels = ['Ground truth', 'Prediction']

            # pred[i, :] = (pred[i, :] - np.min(pred[i, :])) / (np.max(pred[i, :]) - np.min(pred[i, :]))
            cls_pred[i, :] = softmax(cls_pred[i, :])

            l = [[]] #[gt, pred]
            for j in range(cls_gt.shape[1]):
                l[0].append(cls_pred[i, j])
            barWidth = 0.25
            fig, ax = plt.subplots(figsize=(7, 8))
            # ax = fig.add_subplot(111)
            ax.set_ylim((0,1.1))
            # ax.set_xlim((-0.5, 2.5))
            ax.tick_params(axis="y",direction="in")
            ax.tick_params(axis="x",direction="in")

            br = [[]]
            br[0] = np.arange(len(l[0]))
            for j in range(len(l)):
                ax.bar(br[j], l[j], width=barWidth, label=plot_labels[j])

            plt.xticks(range(cls_gt.shape[1]), plot_labels2)
            id_tick_change_colour = np.argmax(cls_gt[i, :])

            plt.setp(ax.get_xticklabels()[id_tick_change_colour], color='red')

            plt.xlabel('Class')
            plt.ylabel('Probability')
            # plt.xticks(rotation=45)
            # plt.legend()
            plt.title('Classification probability')
            fig.canvas.draw()
            # plt.show()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            prob = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 1000 x 1000

            # plot tactile
            tactile = np.where(tactile==0, np.nan, tactile)

            fig = plt.figure(figsize=(4, 4))
            plt.imshow(tactile[i, -1, :, :11], vmin=0, vmax=0.5, cmap='viridis')
            plt.title('Left hand tactile frames')
            plt.axis('off')
            fig.canvas.draw()
            # plt.show()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            tac_plot_left = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
            tac_plot_left = cv2.resize(tac_plot_left, (350, 400))

            fig = plt.figure(figsize=(4, 4))
            plt.imshow(tactile[i, -1, :, 11:], vmin=0, vmax=0.5, cmap='viridis')
            plt.title('Right hand tactile frames')
            plt.axis('off')
            fig.canvas.draw()
            # plt.show()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            tac_plot_right = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
            tac_plot_right = cv2.resize(tac_plot_right, (350, 400))

            tac = np.concatenate((tac_plot_left, tac_plot_right), axis=1) 
            # print (tac.shape)
            img = np.concatenate((curr_frame, tac), axis=0)
            # print (curr_frame.shape)
            img = np.concatenate((img, prob), axis=1)
            # print (prob.shape)
            # # img = np.concatenate((tac, prob), axis=0)

            # # img = video_arr[i, :, :, :]
            # # print (img.shape)
            out.write(img)


##### plot tac #######

if plot_tac:
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_path + name + seq_n + '_tac.avi', fourcc, 30, (1400,800))
    print ('Video streaming')


    for i in range(count_list[0], count_list[-1]+1):
        if i%50 == 0:
            print (i)

        video_ind = np.where(count_list<=i)[0][-1] 
        # print (video_ind)

        if video_path == '03_case/' or video_path == '04_wrench/' or video_path == '05_pitcher_right/':
            # print ('here')
            cam = cv2.VideoCapture('./data/dynamic_examples_raw_data/' + video_clips[video_ind] + '_dyn' + comments[video_ind] + '_aligned.avi')
        else:
            cam = cv2.VideoCapture(main_path + video_path + video_clips[video_ind] + seq_n + comments[v] + '_aligned.avi')
        
        cam.set(1, st_list[video_ind] + i - count_list[video_ind] + 20)
        res, curr_frame = cam.read()
        curr_frame = curr_frame[:200, 350:, :3]
        curr_frame = cv2.resize(curr_frame, (700, 400))

        idx = np.where(tac_gt[i, -20, :, 3, 10] != 0)
        if len(idx[0])>0:
            idx = idx[0][0]
            # print (idx)

            tac_gt[i, -20, idx, :, :] = np.where(tac_gt[i, -20, idx, :, :]==0, np.nan, tac_gt[i, -20, idx, :, :])
            tac_pred[i, -20, idx, :, :] = np.where(tac_pred[i, -20, idx, :, :]==0, np.nan, tac_pred[i, -20, idx, :, :])
        
            fig, ax = plt.subplots(1,3) 
            fig.tight_layout()
            plt.subplots_adjust(wspace=0.05)

            tactile = np.where(tactile==0, np.nan, tactile)

            fig = plt.figure(figsize=(7, 4))
            plt.imshow(np.reshape(tac_gt[i, -20, idx, :, :], (9,22)), vmin=0, vmax=0.5, cmap='viridis')
            plt.title('Ground Truth')
            plt.axis('off')
            fig.canvas.draw()
            # plt.show()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            tac_plot_left = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
            tac_plot_left = cv2.resize(tac_plot_left, (700, 400))

            fig = plt.figure(figsize=(7, 4))
            plt.imshow(np.reshape(tac_pred[i, -20, idx, :, :], (9,22)), vmin=0, vmax=0.5, cmap='viridis')
            plt.title('Prediction')
            plt.axis('off')
            fig.canvas.draw()
            # plt.show()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            tac_plot_right = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
            tac_plot_right = cv2.resize(tac_plot_right, (700, 400))

            prob = np.concatenate((tac_plot_left, tac_plot_right), axis=0) 


            # plot tactile
            tactile = np.where(tactile==0, np.nan, tactile)

            fig = plt.figure(figsize=(4, 4))
            plt.imshow(tactile[i, -1, :, :11], vmin=0, vmax=0.5, cmap='viridis')
            plt.title('Left hand tactile frames')
            plt.axis('off')
            fig.canvas.draw()
            # plt.show()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            tac_plot_left = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
            tac_plot_left = cv2.resize(tac_plot_left, (350, 400))

            fig = plt.figure(figsize=(4, 4))
            plt.imshow(tactile[i, -1, :, 11:], vmin=0, vmax=0.5, cmap='viridis')
            plt.title('Right hand tactile frames')
            plt.axis('off')
            fig.canvas.draw()
            # plt.show()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            tac_plot_right = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
            tac_plot_right = cv2.resize(tac_plot_right, (350, 400))

            tac = np.concatenate((tac_plot_left, tac_plot_right), axis=1) 
            # print (tac.shape)
            img = np.concatenate((curr_frame, tac), axis=0)
            # print (curr_frame.shape)
            img = np.concatenate((img, prob), axis=1)
            # print (prob.shape)
            # # img = np.concatenate((tac, prob), axis=0)

            # # img = video_arr[i, :, :, :]
            # # print (img.shape)
            out.write(img)


######  plot dyn ########
if plot_dyn:
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_path + name + seq_n + '_dyn_2.avi', fourcc, 30, (1400,800))
    print ('Video streaming')


    for i in range(count_list[0], count_list[-1]+1):
        if i%50 == 0:
            print (i)

        video_ind = np.where(count_list<=i)[0][-1] 
        # print (video_ind)

        if video_path == '03_case/' or video_path == '04_wrench/' or video_path == '05_pitcher_right/':
            # print ('here')
            cam = cv2.VideoCapture('./data/dynamic_examples_raw_data/' + video_clips[video_ind] + '_dyn' + comments[video_ind] + '_aligned.avi')
        else:
            cam = cv2.VideoCapture(main_path + video_path + video_clips[video_ind] + seq_n + comments[v] + '_aligned.avi')
        
        cam.set(1, st_list[video_ind] + i - count_list[video_ind])
        res, curr_frame = cam.read()
        curr_frame = curr_frame[:200, 350:, :3]
        curr_frame = cv2.resize(curr_frame, (700, 400))

        # plot dynamics

        fig, ax = plt.subplots(figsize=(7, 8))
        ax.set_ylim((0,dyn_range+1000))

        # plt.plot(smooth(dyn_gt[:, 1, dyn_index] * dyn_range, 5))
        # plt.plot(smooth(dyn_pred[:, 1, dyn_index] * dyn_range, 5))
        # plt.show()
        
        plt.plot(smooth(dyn_gt[i:i+40, 1, dyn_index] * dyn_range, 10), linewidth=5, label='gt')
        plt.plot(smooth(dyn_pred[i:i+40, 1, dyn_index] * dyn_range, 10), linewidth=5, label='pred')
        # plt.plot(smooth(dyn_pred[i:i+40, 1, dyn_index] * dyn_range * 0.8, 10), linewidth=5, label='pred')
        plt.legend()
        plt.title('Object dynamics estimation')
        plt.xlabel('Time step')
        # plt.ylabel('Rotation angle (degree)')
        plt.ylabel('Weight (g)')
        fig.canvas.draw()
        # plt.show()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        prob = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 1000 x 1000

        # plot tactile
        tactile = np.where(tactile==0, np.nan, tactile)

        fig = plt.figure(figsize=(4, 4))
        plt.imshow(tactile[i, -1, :, :11], vmin=0, vmax=0.5, cmap='viridis')
        plt.title('Left hand tactile frames')
        plt.axis('off')
        fig.canvas.draw()
        # plt.show()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        tac_plot_left = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
        tac_plot_left = cv2.resize(tac_plot_left, (350, 400))

        fig = plt.figure(figsize=(4, 4))
        plt.imshow(tactile[i, -1, :, 11:], vmin=0, vmax=0.5, cmap='viridis')
        plt.title('Right hand tactile frames')
        plt.axis('off')
        fig.canvas.draw()
        # plt.show()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        tac_plot_right = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 500
        tac_plot_right = cv2.resize(tac_plot_right, (350, 400))

        tac = np.concatenate((tac_plot_left, tac_plot_right), axis=1) 
        # print (tac.shape)
        img = np.concatenate((curr_frame, tac), axis=0)
        # print (curr_frame.shape)
        img = np.concatenate((img, prob), axis=1)
        # print (prob.shape)
        # # img = np.concatenate((tac, prob), axis=0)

        # # img = video_arr[i, :, :, :]
        # # print (img.shape)
        out.write(img)


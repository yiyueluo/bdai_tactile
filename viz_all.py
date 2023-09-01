import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import cv2
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

label_list = ['picking up (T0-0)', 'inserting (T0-1)', 'threading (T0-2)', 'releasing (T0-3)', 
'picking up (T1-0)', 'inserting (T1-1)', 'tightening (T1-2)', 'releasing (T1-3)', 'putting down (T1-4)', 
'unlocking right (T2-0)', 'unlocking left (T2-1)', 'lifting (T2-2)', 'releasing (T2-3)', 
'picking up (T3-0)', 'rotating left (T3-1)', 'straightening from left (T3-2)', 'rotating right (T3-3)', 'straightening from left (T3-4)', 'releasing (T3-5)',
'picking up (T4-0)', 'pouring (T4-1)', 'straightening (T4-2)', 'putting down (T4-3)']
color = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

main_path = './dataset/all_dominant/predictions/'
name = 'eval_exp22'
# name = 'eval_exp24_01_unknown'
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

label_count = [4, 5, 4, 6, 4]


# plot label 
cls_gt = label_gt[1:, -1, :] * label_mask[1:, -1, :]
cls_pred = label_pred[1:, -1, :] * label_mask[1:, -1, :]
# print (cls_gt.shape, cls_pred.shape)

# fig, ax = plt.subplots(1,2) 
# ax[0].imshow(cls_gt,  aspect=0.01)
# ax[1].imshow(cls_pred,  aspect=0.01)
# plt.show()

# # plot confusion matrix
# for i in range(len(label_count)):
confusion_gt = np.argmax(cls_gt, axis=1)
confusion_pred = np.argmax(cls_pred, axis=1)

a = accuracy_score(confusion_gt, confusion_pred[:])
print('accuracy:', a)

cm = confusion_matrix(confusion_gt, confusion_pred, labels=np.arange(23), normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
fig, ax = plt.subplots(figsize=(8, 8))
# font = {'family' : 'normal',
#     'size'   : 4}
# plt.rc('font', **font)
disp.plot(ax=ax, xticks_rotation = 'vertical')
plt.show()


# individual task viz
c = 0
for i in range(len(label_count)):
    gt_t = []
    pred_t = []
    for j in range(confusion_gt.shape[0]):
        if confusion_gt[j] >= c and confusion_gt[j] < c + label_count[i]:
            gt_t.append(confusion_gt[j])
            pred_t.append(confusion_pred[j])
    
    a = accuracy_score(gt_t, pred_t[:])
    print('accuracy:', i, a, c, c + label_count[i])
    # print (np.arange(c, c+label_count[i]))

    # cm = confusion_matrix(gt_t, pred_t, labels=np.arange(c, c+label_count[i]), normalize='true')
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list[c:c + label_count[i]])
    # fig, ax = plt.subplots(figsize=(8, 8))
    # disp.plot(ax=ax, xticks_rotation = 'vertical')
    # plt.show()

    c += label_count[i]





# plot tsne
print (feature.shape)
tsne = TSNE(n_components=2, perplexity=50, n_iter=5000) #50
x = tsne.fit_transform(feature[1:, :]) 
label_count = [4, 5, 4, 6, 4]
task_label = ['bolt', 'screw', 'case', 'wrench', 'pitcher']

fig = plt.figure(figsize=(5,5))
axis = fig.add_subplot(111, projection='3d')

c = 0
for n, l in enumerate(label_count):
    l_gt = confusion_pred - c
    l_gt = np.where(l_gt < 0, 100, l_gt)
    ix = np.where(l_gt < l)
    c += l
    axis.scatter(x[ix,0],x[ix,1], c = color[n], label = task_label[n])
    # axis.scatter(x[ix,0],x[ix,1],x[ix,2], c = color[n], label = task_label[n])

# for n, l in enumerate(label_list):
# for n in range(13, 19):
#     ix = np.where(confusion_gt == n)
#     axis.scatter(x[ix,0],x[ix,1],x[ix,2], c = color[n-13], label = label_list[n])

# axis.scatter(x[:,0],x[:,1],x[:,2])
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
# axis.set_zlabel("PC3", fontsize=10)
axis.legend()
plt.show()



# plot dyn
if dyn_pred.shape[0] > 1:
    dyn_gt = dyn_gt * dyn_mask
    dyn_pred = dyn_pred * dyn_mask
    dyn_gt_to_plot = np.zeros((dyn_gt.shape[0]))
    dyn_pred_to_plot = np.zeros((dyn_gt.shape[0]))

    # mse = np.mean(np.abs((dyn_gt[:, 1, 0] * 90 - dyn_pred[:, 1, 0] * 90)))
    # print (mse)

    # plt.plot(dyn_gt[:, 1, 0], label='gt')
    # plt.plot(dyn_pred[:, 1, 0], label='pred')
    # plt.legend()
    # plt.show()

    mse3 = np.mean(np.abs((dyn_gt[3600:4400, 1, 2] * 90 - dyn_pred[3600:4400, 1, 2] * 90)))
    mse4 = np.mean(np.abs((dyn_gt[4000:4800, 1, 3] * 180 - dyn_pred[4000:4800, 1, 3] * 180)))
    mse5 = np.mean(np.abs((dyn_gt[5200:, 1, 4] * 1400 - dyn_pred[5200:, 1, 4] * 1400)))
    print (mse3, mse4, mse5)

    
    plt.plot(smooth(dyn_gt[:, 1, 2] * 100, 5), label='gt')
    plt.plot(smooth(dyn_pred[:, 1, 2] * 100, 5), label='pred')
    plt.legend()
    plt.show()

    plt.plot(smooth(dyn_gt[: 1, 3], 10) * 180, label='gt')
    plt.plot(smooth(dyn_pred[:, 1, 3], 10) * 180, label='pred')
    plt.legend()
    plt.show()

    plt.plot(smooth(dyn_gt[5000:, 1, 4], 10) * 1400, label='gt') # 1580g for scale, x1400
    plt.plot(smooth(dyn_pred[5000:, 1, 4], 10) * 1400, label='pred')
    plt.legend()
    plt.show()
    # for i in range(1, dyn_gt.shape[0]):
    #     if np.sum(dyn_gt[i, 1, :]) != 0:
    #     # print (dyn_gt[i, 1, :])
    #         idx = np.where(dyn_gt[i, 1, :] != 0)[0]
    #         print (idx)
    #         dyn_gt_to_plot[i] = dyn_gt[i, -1, idx]
    #         dyn_pred_to_plot[i] = dyn_pred[i, -1, idx]
    # plt.plot(dyn_gt_to_plot, label='gt')
    # plt.plot(dyn_pred_to_plot, label='pred')
    # plt.legend()
    # plt.show()


# plot tac img
if tac_pred.shape[0] > 1:
    tac_gt = tac_gt * tac_mask
    tac_pred = tac_pred * tac_mask


    for i in range(5):
        mse = np.mean(np.abs(tac_pred[:, -20, i, :, :] * 40 - tac_gt[:, -20, i, :, :] * 40))
        print ('gt diff:', mse)
    # print ('input diff:', np.sum((tac_pred[:, -20, :, :] - tac_input[:, -1, :, :])**2))
    # mpl.use('Agg')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(main_path + name + '.avi', fourcc, 5, (640,480))
    print ('Video streaming')
    for i in range(4360, tac_gt.shape[0]):
        if i%50 == 0:
            print (i, tac_gt.shape[0])

        idx = np.where(tac_gt[i, -20, :, 5, 10] != 0)
        # if len(idx) != 1:
        # print (idx)

        tac_gt[i, -20, idx, :, :] = np.where(tac_gt[i, -20, idx, :, :]==0, np.nan, tac_gt[i, -20, idx, :, :])
        tac_pred[i, -20, idx, :, :] = np.where(tac_pred[i, -20, idx, :, :]==0, np.nan, tac_pred[i, -20, idx, :, :])
        tac_input[i, -1, :, :] = np.where(tac_input[i, -1, :, :]==0, np.nan, tac_input[i, -1, :, :])
        
        fig, ax = plt.subplots(1,3) 
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.05)

        ax[0].imshow(tac_input[i, -1, :, :],  vmin=0, vmax=0.3)
        ax[1].imshow(np.reshape(tac_gt[i, -20, idx, :, :], (9,22)),  vmin=0, vmax=0.3)
        ax[2].imshow(np.reshape(tac_pred[i, -20, idx, :, :], (9,22)),  vmin=0, vmax=0.3)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        plt.show()


        # fig.canvas.draw()
        # frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 480 x 640
        # img2 = img.copy()
        # # print (img2.shape)
        # # cv2.putText(img2, 'label:' + str(label_index[i]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 2) 
        # cv2.putText(img2, 'frame:' + str(i), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 2) 

        # out.write(img2)
        # plt.clf()
        # plt.close(fig)



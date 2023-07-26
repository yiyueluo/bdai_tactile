import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# names = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# names = ['40', 'thumb', 'index', 'middle', 'ring', 'little', 'palm', 'fingers', 'fingertips', 'fingermids']
# names = ['40', 'sub2', 'sub4', 'sub8']
# names = ['40', 'cross1', 'cross2', 'cross3', 'cross4', 'cross5', 'cross6', 'cross7', 'cross8', 'cross9']
# names = ['tsne', 'left2']
names = ['']


acc_list = []

main_path = './dataset/03_case/predictions/'
# data = pickle.load(open(main_path + 'train_01_bolt.p', "rb")) # train
data = pickle.load(open(main_path + 'train_best_03_case.p', "rb")) # train
# data = pickle.load(open(main_path + 'eval_03_case.p', "rb")) # val 
# data = pickle.load(open(main_path + 'val_best_03_case.p', "rb")) # val 

for n in names:

    # data = pickle.load(open(main_path + 'eval_03_case_' + str(n) + '.p', "rb")) # test

    # tac_left = data[0]
    # tac_right = data[1]
    tactile = data[0]
    gt = data[1]
    pred = data[2]

    # print (gt.shape, pred.shape)

    # gt = np.delete(gt, [2, 3], 1)
    # pred = np.delete(pred, [2, 3], 1)

    gt = np.delete(gt, [2], 1)
    pred = np.delete(pred, [2], 1)
    

    # gt = gt[:8000, :]
    # pred = pred[:8000, :]

    pred = softmax(pred, axis=1)

    fig, ax = plt.subplots(1,2) 
    ax[0].imshow(gt,  aspect=0.01)
    ax[1].imshow(pred,  aspect=0.01)
    plt.show()


    # name = ['picking up bolt', 'inserting bolt', 'empty insertion', 'mismath insertion', 'threading', 'releasing' ]
    # name = ['picking up tool', 'inserting tool', 'mismath insertion', 'threading', 'taking out tool', 'releasing' ]
    name = ['picking up bolt', 'inserting bolt', 'threading', 'releasing' ]

    y_true = np.argmax(gt, axis=1)
    predictions = np.argmax(pred, axis=1)

    # unique, counts = np.unique(y_true, return_counts=True)
    # print (dict(zip(unique, counts)))
    # unique, counts = np.unique(predictions, return_counts=True)
    # print (dict(zip(unique, counts)))

    a = accuracy_score(y_true, predictions)
    print(n, 'accuracy:', a)
    acc_list.append(a)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1, 2, 3], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=name)
    disp.plot()
    plt.show()


    # for i in range(tactile.shape[0]):
    #     plt.imshow(tactile[i, 0, :, :])
    #     print ('pred stage:', np.argmax(pred[i, :]), 'gt stage:', np.argmax(gt[i, :]))
    #     plt.show()

plt.plot(acc_list)
plt.ylabel('accuracy')
plt.xlabel('blocking area')
# plt.xticks([0, 1, 2, 3], ['sub1', 'sub2', 'sub4', 'sub8'])
plt.xticks(np.arange(len(names)), names)
plt.show()
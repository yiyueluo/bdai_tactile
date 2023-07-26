import scipy.io
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from utils import *


def pca(tac, label, viz):
    # Scale data before applying PCA
    scaling=StandardScaler()
    scaling.fit(tac)
    Scaled_data=scaling.transform(tac)

    principal=PCA(n_components=2)
    principal.fit(Scaled_data)
    x = principal.transform(Scaled_data)

    if viz:
        plt.figure(figsize=(5,5))
        plt.scatter(x[:,0],x[:,1],c=label,cmap='plasma')
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.show()


        # fig = plt.figure(figsize=(10,10))
        # axis = fig.add_subplot(111, projection='3d')
        # axis.scatter(x[:,0],x[:,1],x[:,2], c=label,cmap='plasma')
        # axis.set_xlabel("PC1", fontsize=10)
        # axis.set_ylabel("PC2", fontsize=10)
        # axis.set_zlabel("PC3", fontsize=10)
        # plt.show()



def tsne(tac, label, viz):
    # scaling=StandardScaler()
    # scaling.fit(tac)
    # Scaled_data=scaling.transform(tac)

    tsne = TSNE(n_components=2, perplexity=50, n_iter=5000)
    x = tsne.fit_transform(tac) 

    color = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    labels = ['pick up', 'insert', 'empty', 'mismatch', 'thread', 'realease']
    # labels = ['pick up', 'insert', 'thread', 'realease']
    
    if viz:
        fig, ax = plt.subplots()
        for g in np.unique(label):
            ix = np.where(label == g)
            ax.scatter(x[ix,0],x[ix,1], c = color[g], label = labels[g])
        ax.legend()
        plt.show()


        # fig = plt.figure(figsize=(5,5))
        # axis = fig.add_subplot(111, projection='3d')
        # for g in np.unique(label):
        #     ix = np.where(label == g)
        #     axis.scatter(x[ix,0],x[ix,1],x[ix,2], c=color[g], label = labels[g])
        # axis.set_xlabel("PC1", fontsize=10)
        # axis.set_ylabel("PC2", fontsize=10)
        # axis.set_zlabel("PC3", fontsize=10)
        # axis.legend()
        # plt.show()


def dbscan(tac, label, viz):
    scaling=StandardScaler()
    scaling.fit(tac)
    Scaled_data=scaling.transform(tac)
    db = DBSCAN(eps=0.02, min_samples=10).fit(Scaled_data)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    


viz = True

# cluster on tactile images
# path = './dataset/01_bolt/train/'
# log = np.asanyarray(pickle.load(open(path + 'log.p', "rb")))

# tac_left = np.zeros((1, 9, 11))
# tac_right = np.zeros((1, 9, 11))
# label = np.zeros((1, 6))
# for f in range(log.shape[0]-1):
#     local_path = path + str(log[f]) + '.p'
#     data = pickle.load(open(local_path, "rb"))
#     tac_left = np.concatenate((tac_left, data[0]), axis=0)
#     tac_right = np.concatenate((tac_right, data[1]), axis=0)
#     label = np.concatenate((label, data[2]), axis=0)

# label = np.argmax(label[1: :], axis=1)
# tac = np.concatenate((tac_left[1:, :, :], tac_right[1:, :, :]), axis=1)
# tac = np.reshape(tac, (tac.shape[0], -1))

# tsne(tac, label, viz)
# # pca(tac, label, viz)


# cluster on features
main_path = './dataset/01_bolt/predictions/'
data = pickle.load(open(main_path + 'eval_01_bolt_tsne.p', "rb")) # test

tac = data[0]
gt = data[1]
pred = data[2]
feature = data[3]

label = np.argmax(gt, axis=1)

tsne(feature[2500:, :], label[2500:], viz)
# tsne(feature[:800, :], label[:800], viz)
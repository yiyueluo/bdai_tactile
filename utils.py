import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pickle

def read_video(path):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print ('fps:', fps, '# frames:', n_frames)
    return fps, n_frames


def read_tactile_csv(path):
    df = pd.read_csv(path, sep=',', header=0)
    data = df.to_numpy()
    ts = data[:, 0]
    pressure = data[:, 1:]
    fps = 1 / (data[1, 0] - data[0, 0])

    print ('fps:', fps, '# frames:', ts.shape[0])
    return fps, ts, pressure



def align_ts(tac_marker_frame, videos_marker_frame, tac_ts, videos_ts):
    videos_aligned_ts = []
    videos_aligned_index = []
    n_cam = len(videos_marker_frame)
    # print (n_cam)
    for n in range(n_cam):
        v_marker_frame = videos_marker_frame[n]
        v_ts = videos_ts[n]
    
        t_diff = v_ts[v_marker_frame] - tac_ts[tac_marker_frame]
        v_ts = v_ts - t_diff

        v_aligned_ts = np.copy(tac_ts)
        v_aligned_index = np.copy(tac_ts)
        for i in range(tac_ts.shape[0]):
            index = np.argmin(np.abs(v_ts - tac_ts[i]))
            # print (index)
            v_aligned_ts[i] = v_ts[index]
            v_aligned_index[i] = index
    
        videos_aligned_ts.append(v_aligned_ts)
        videos_aligned_index.append(v_aligned_index)

    return videos_aligned_ts, videos_aligned_index 


def export_tac(tac, tac_layout, save_path):
    arr = np.zeros((tac.shape[0], 9, 11)) # frames x 9 x 11
    for f in range(tac.shape[0]):
        tac_frame = tac[f, :]
        for i in range(tac_layout.shape[0]):
            if tac_layout[i, 2] != -1:
                arr[f, tac_layout[i, 0], tac_layout[i, 1]] = tac_frame[tac_layout[i, 2]]

    pickle.dump(arr, open(save_path, "wb"))
    print ('dumped', save_path, arr.shape)



def viz_tac(tac, tac_layout, viz):
    arr = np.zeros((9, 11)) #viz 9x11
    for i in range(tac_layout.shape[0]):
        if tac_layout[i, 2] != -1:
            arr[tac_layout[i, 0], tac_layout[i, 1]] = tac[tac_layout[i, 2]]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.imshow(arr, vmin=0, vmax=1, cmap='viridis')
    if viz:
        plt.show()

    fig.canvas.draw()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    tac_img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    tac_img = cv2.resize(tac_img, (100, 200))

    if viz:
        cv2.imshow('img', tac_img)
        cv2.waitKey(1)
    
    plt.close(fig)
    plt.clf()

    return tac_img


def viz(tac, videos_path, videos_aligned_index, tac_layout, st, save_path, viz):
    plt.ioff()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_path, fourcc, 30, (700,400))
    print ('Video streaming')

    cam0 = cv2.VideoCapture(videos_path[0])
    cam1 = cv2.VideoCapture(videos_path[1])

    for n_frame in range(st, tac.shape[0]):
        if n_frame % 500 == 0:
            print (str(n_frame) + '/' + str(tac.shape[0]))

        fig = plt.figure(figsize=(5, 2))
        ax = fig.add_subplot(111)
        ax.set_ylim(0, np.amax(tac))
        ax.set_xlim(0, tac.shape[0])
        for n_sensor in range(tac.shape[1]):
            ax.plot(tac[:n_frame, n_sensor])

        fig.canvas.draw()
        # plt.show()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        tac_plot = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 200

        left = viz_tac(tac[n_frame, :], tac_layout[0], viz=False)
        right = viz_tac(tac[n_frame, :], tac_layout[1], viz=False)
        tac_map = np.concatenate((left, right), axis=1) # 200 x 200

        tac_img = np.concatenate((tac_map, tac_plot), axis=1) # 700 x 200
        # print (tac_plot.shape, tac_map.shape, tac_img.shape)

        cam0.set(1, videos_aligned_index[0][n_frame])
        res, curr_frame = cam0.read()
        cam0_img = cv2.resize(curr_frame, (350, 200))
        # cv2.imshow('image', cam0_img)

        cam1.set(1, videos_aligned_index[1][n_frame])
        res, curr_frame = cam1.read()
        cam1_img = cv2.resize(curr_frame, (350, 200))
        # cv2.imshow('image', cam1_img)

        img = np.concatenate((cam0_img, cam1_img), axis=1)
        img = np.concatenate((img, tac_img), axis=0)

        cv2.putText(img, 'frame:' + str(n_frame), (10, 210), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3) 

        if viz:
            cv2.imshow('img', img)
            cv2.waitKey(1)

        out.write(img)
        plt.clf()
        plt.close(fig)

    exit(0)



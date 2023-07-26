import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import time
import h5py
import math

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def read_video(path):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print ('video fps:', fps, '# frames:', n_frames)
    return fps, n_frames


def read_tactile_csv(path):
    df = pd.read_csv(path, sep=',', header=0)
    data = df.to_numpy()
    ts = data[:, 0]
    pressure = data[:, 1:]
    fps = 1 / (data[1, 0] - data[0, 0])

    print ('video fps:', fps, '# frames:', ts.shape[0])
    return fps, ts, pressure

    
def convert_time(timestring):
    pt = datetime.strptime(timestring,'%H:%M:%S:%f')
    # print (pt.microsecond, pt.second, pt.minute, pt.hour)
    total_seconds = pt.microsecond/1000000 + pt.second + pt.minute*60 + pt.hour*3600 
    # print (total_seconds)

    return total_seconds


def read_tactile_csv_with_timestamp(path):
    df = pd.read_csv(path, sep=',', header=6)
    data = df.to_numpy()
    ts = data[:, 0]
    pressure = data[:, 1:]
    fps = 1 / (data[1, 0] - data[0, 0])

    print ('tactile fps:', fps, '# frames:', ts.shape[0])
    return fps, ts, pressure

def read_mocap(path, n_body):
    df = pd.read_csv(path, sep=',', header=6)
    data = df.to_numpy()
    ts = data[:, 1]
    mocap_data = []
    for n in range(n_body):
        # x = []
        # y = []
        # z = []
        # for i in range(data.shape[0]):
        #     # a = euler_from_quaternion(data[i, 18 + 19 * n], data[i, 19 + 19 * n], data[i, 20 + 19 * n], data[i, 21 + 19 * n])
        #     # x.append(a[0])
        #     # y.append(a[1])
        #     # z.append(a[2])

        #     x.append(data[i, 18 + 19 * n])
        #     y.append(data[i, 18 + 19 * n])
        #     z.append(data[i, 18 + 19 * n])

        # mocap_data.append([np.asanyarray(x), np.asanyarray(y), np.asanyarray(z)])
        mocap_data.append([data[:, 18 + 19 * n], data[:, 19 + 19 * n], data[:, 20 + 19 * n]])
    fps = 1 / (data[1, 1] - data[0, 1])

    # plt.plot(mocap_data[0][0])
    # plt.show()

    print ('mocap fps:', fps, '# frames:', ts.shape[0])
    return fps, ts, mocap_data


def read_scale(path):
    f = h5py.File(path, 'r')
    fc = f['frame_count'][0]
    ts = np.array(f['ts'][:fc])
    scale = np.array(f['scale'][:fc]).astype(np.float32)
    fps = 1 / (ts[3] - ts[2])
    # remove lost dataframe
    scale = np.where(scale<1e6, 0, scale)
    n = np.min(scale)
    while n==0:
        for i in range(1, len(scale)):
            if scale[i] == 0:
                scale[i] = scale[i-1]

        for i in range(len(scale)-1):
            if scale[i] == 0:
                scale[i] = scale[i+1]        
        n = np.min(scale)
    scale = smooth(scale, 2)

    print(fc, ts, scale)
    print ('fps:', fps, '# frames:', ts.shape[0])

    return ts, scale

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


def align_ts_with_timestamp(tac_ts, to_align_ts):
    align_ts_index = []
    for v in range(len(to_align_ts)): 
        align_ts_index_v = []
        for i in range(tac_ts.shape[0]):
            index = np.argmin(np.abs(tac_ts[i] - to_align_ts[v]))
            align_ts_index_v.append(index)
        align_ts_index.append(align_ts_index_v)
        # print (align_ts_index_v)
    
    return align_ts_index


def export_tac(tac, tac_layout, save_path):
    arr = np.zeros((tac.shape[0], 9, 11)) # frames x 9 x 11
    for f in range(tac.shape[0]):
        tac_frame = tac[f, :]
        for i in range(tac_layout.shape[0]):
            if tac_layout[i, 2] != -1:
                arr[f, tac_layout[i, 0], tac_layout[i, 1]] = tac_frame[tac_layout[i, 2]]

    pickle.dump(arr, open(save_path, "wb"))
    print ('dumped', save_path, arr.shape)


def export_mocap(mocap_data, mocap_aligned_index, save_path):
    to_save = []
    for i in range(len(mocap_data)):
        rigid = []
        for j in range(3):
            rigid.append(mocap_data[i][j][mocap_aligned_index])
        to_save.append(rigid)
    pickle.dump(to_save, open(save_path, "wb"))


def export_scale(scale_data, scale_aligned_index, save_path):
    to_save = scale_data[scale_aligned_index]
    pickle.dump(to_save, open(save_path, "wb"))


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

def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])
    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),colors)):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)
        axis.label.set_color(c)
        axis.line.set_color(c)
        axis.set_tick_params(colors=c)
        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]
        ax.text(*text_plot, axlabel.upper(), color=c, va="center", ha="center")
    ax.text(*offset, name, color="k", va="center", ha="center", bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})


def viz(tac, videos_path, videos_aligned_index, tac_layout, st, save_path, viz):
    plt.ioff()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_path, fourcc, 30, (700,400))
    print ('Video streaming')

    cam0 = cv2.VideoCapture(videos_path[0])
    cam1 = cv2.VideoCapture(videos_path[1])

    for n_frame in range(st-50, tac.shape[0]):
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


def viz_static_with_timestamp(tac, videos_path, videos_aligned_index, tac_layout, save_path, viz):
    plt.ioff()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_path, fourcc, 30, (700,400))
    print ('Video streaming')

    cam0 = cv2.VideoCapture(videos_path[0])
    cam1 = cv2.VideoCapture(videos_path[1])

    for n_frame in range(300, tac.shape[0]):
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

        cv2.putText(img, 'tac frame:' + str(n_frame), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 2) 
        cv2.putText(img, 'cam0 frame:' + str(aligned_index[0][n_frame]), (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 2) 
        cv2.putText(img, 'cam1 frame:' + str(aligned_index[1][n_frame]), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 2)


        if viz:
            cv2.imshow('img', img)
            cv2.waitKey(1)

        out.write(img)
        plt.clf()
        plt.close(fig)

    exit(0)

def viz_dynamic_with_timestamp(tac, videos_path, aligned_index, aligned_dynamic_index, tac_layout, dynamic_data, aligned_dynamic_label, save_path, viz):
    plt.ioff()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_path, fourcc, 30, (700,600))
    print ('Video streaming')

    cam0 = cv2.VideoCapture(videos_path[0])
    cam1 = cv2.VideoCapture(videos_path[1])

    for n_frame in range(0, tac.shape[0], 1):
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

        cam0.set(1, aligned_index[0][n_frame])
        res, curr_frame = cam0.read()
        cam0_img = cv2.resize(curr_frame, (350, 200))
        # cv2.imshow('image', cam0_img)

        cam1.set(1, aligned_index[1][n_frame])
        res, curr_frame = cam1.read()
        cam1_img = cv2.resize(curr_frame, (350, 200))
        # cv2.imshow('image', cam1_img)

        img = np.concatenate((cam0_img, cam1_img), axis=1)
        img = np.concatenate((img, tac_img), axis=0)

        # plot dynamic parameters
        fig = plt.figure(figsize=(5, 2)) # 500 x 200
        ax = fig.add_subplot(111)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, tac.shape[0])
        ax.plot(dynamic_data[0][:n_frame])
        # ax.plot(dynamic_data[1][:n_frame])

        for n in range(len(dynamic_data)):
            ax.plot(dynamic_data[n][:n_frame], label=aligned_dynamic_label[n])
        ax.legend()


        fig.canvas.draw()
        # plt.show()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        sq = np.zeros((200, 200, 3), dtype=np.uint8)
        sq.fill(255)
        dynamic_plot = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1] # 500 x 200
        dynamic_plot = np.concatenate((sq, dynamic_plot), axis=1) # 700 x 200
        img = np.concatenate((img, dynamic_plot), axis=0)


        cv2.putText(img, 'tac frame:' + str(n_frame), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 2) 
        cv2.putText(img, 'cam0 frame:' + str(aligned_index[0][n_frame]), (10, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 2) 
        cv2.putText(img, 'cam1 frame:' + str(aligned_index[1][n_frame]), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 2)

        for n in range(len(aligned_dynamic_index)):
            cv2.putText(img, 'dyn frame:' + str(aligned_dynamic_index[n][n_frame]), (10, 455+15*n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 2) 

        if viz:
            cv2.imshow('img', img)
            cv2.waitKey(1)

        out.write(img)
        plt.clf()
        plt.close(fig)

    exit(0)
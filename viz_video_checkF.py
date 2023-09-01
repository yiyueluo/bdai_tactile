import numpy as np
import scipy.io



video_path = './data/processed/02_screw/'
video_clips = ['rec04_02']
rd = [6] 
st_frame = [645]
count_list = [0]
st_list = []

count = 0
# for v in range(1):
for v in range(len(video_clips)):
    label_path = video_path + video_clips[v] +  '_arr.mat'
    mat = scipy.io.loadmat(label_path)
    label_data = mat['LabelData']
    extracted_st = (np.where(label_data[:, 0]==1))[0][rd[v]-1]  # round -1 for index
    extracted_ed = (np.where(label_data[:, -1]==1))[0][rd[v]-1]

    count = count + (extracted_ed - extracted_st)
    count_list.append(count)
    st_list.append(extracted_st)

count_list = np.asanyarray(count_list)
print (count_list)

from utils import *

main_path = './data/'
tac_fps, tac_ts, tac_data = read_tactile_csv(main_path + '0616_rec02.csv')

tac_layout_left_path = main_path + 'common/hand_layout_left.csv'
tac_layout_right_path = main_path + 'common/hand_layout_right.csv'

df = pd.read_csv(tac_layout_left_path, sep=',', header=0)
tac_layout_left = df.to_numpy() #[index 0, index 1, element no.]

df = pd.read_csv(tac_layout_right_path, sep=',', header=0)
tac_layout_right = df.to_numpy() #[index 0, index 1, element no.]

for n_frame in range(6000, tac_data.shape[0], 200):
    frame = tac_data[n_frame, :]
    tac_left = viz_tac(frame, tac_layout_left, viz=True)
    tac_right = viz_tac(frame, tac_layout_right, viz=True)
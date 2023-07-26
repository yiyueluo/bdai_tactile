import math
from scipy.spatial.transform import Rotation as R
from utils import *

def calculate_angle(a1, b1, c1, a2, b2, c2):
     
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    
    return A


import math
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
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



# case [xyzw] --> 18, 19, 20, 21 --> z
# pitcher [xyzw] --> 37, 38, 39, 40 --> x
# wrench [xyzw] --> 56, 57, 58, 59 --> y, z(90)

path = './data/0725_rec01_dyn_mocap.csv'
df = pd.read_csv(path, sep=',', header=6)
data = df.to_numpy()
print (data.shape)

x = []
y = []
z = []
# for i in range(data.shape[0]):
#     r = R.from_quat([data[i, 18 + 19*3], data[i, 19 + 19*3], data[i, 20+ 19*3], data[i, 21+ 19*3]])
#     # r = R.from_quat([data[i, 2], data[i, 3], data[i, 4], data[i, 5]])
#     # r = R.from_quat([data[i,37], data[i, 38], data[i, 39], data[i, 40]])
#     # r = R.from_quat([data[i, 56], data[i, 57], data[i, 58], data[i, 59]])
#     a = r.as_euler('xyz', degrees=True)
#     # a = euler_from_quaternion(data[i, 18], data[i, 19], data[i, 20], data[i, 21])
#     # a = euler_from_quaternion(data[i,37], data[i, 38], data[i, 39], data[i, 40])
#     # a = euler_from_quaternion(data[i, 56], data[i, 57], data[i, 58], data[i, 59])


#     # x.append(a[0])
#     # y.append(a[1])
#     # z.append(a[2])

#     # x.append(data[i, 18 + 19*5])
#     # y.append(data[i, 19 + 19*5])
#     # z.append(data[i, 20 + 19*5])


    
# fig, axs = plt.subplots(3)
# axs[0].plot(x)
# axs[1].plot(y)
# axs[2].plot(z)

# plt.show()

# ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
# for i in range(20000, len(x), 100):
#     # print (i)
#     r = R.from_euler("xyz", [x[i], y[i], z[i]], degrees=True)
#     ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
#     plot_rotated_axes(ax, r, name="r0", offset=(0, 0, 0))
# plt.show()

for i in range(6):
    fig, axs = plt.subplots(3)
    axs[0].plot(data[:, 18 + 19*i])
    axs[1].plot(data[:, 19 + 19*i])
    axs[2].plot(data[:, 20 + 19*i])

    plt.show()

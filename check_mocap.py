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

# case [xyzw] --> 18, 19, 20, 21 --> z
# pitcher [xyzw] --> 37, 38, 39, 40 --> x
# wrench [xyzw] --> 56, 57, 58, 59 --> y, z(90)

path = './data/test/Take 2023-07-11 01.31.53 PM.csv'
df = pd.read_csv(path, sep=',', header=6)
data = df.to_numpy()
print (data.shape)

x = []
y = []
z = []
for i in range(data.shape[0]):
    # r = R.from_quat([data[i, 18], data[i, 19], data[i, 20], data[i, 21]])
    # r = R.from_quat([data[i,37], data[i, 38], data[i, 39], data[i, 40]])
    # r = R.from_quat([data[i, 56], data[i, 57], data[i, 58], data[i, 59]])
    # a = r.as_euler('XYZ', degrees=True)
    # a = euler_from_quaternion(data[i, 18], data[i, 19], data[i, 20], data[i, 21])
    # a = euler_from_quaternion(data[i,37], data[i, 38], data[i, 39], data[i, 40])
    a = euler_from_quaternion(data[i, 56], data[i, 57], data[i, 58], data[i, 59])

    x.append(a[0])
    y.append(a[1])
    z.append(a[2])

    # x.append(data[i, 56])
    # y.append(data[i, 57])
    # z.append(data[i, 58])

    

fig, axs = plt.subplots(3)
axs[0].plot(x)
axs[0].plot(y)
axs[0].plot(z)

plt.show()
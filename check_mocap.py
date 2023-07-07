import math
from utils import *

def calculate_angle(a1, b1, c1, a2, b2, c2):
     
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    
    return A

path = './data/mocap-trial-1.csv'
df = pd.read_csv(path, sep=',', header=0)
data = df.to_numpy()

x1 = data[:, -6]
y1 = data[:, -5]
z1 = data[:, -4]

x2 = data[:, -3]
y2 = data[:, -2]
z2 = data[:, -1]

a_list = []
for i in range(data.shape[0]):
    x = x1[i] - x2[i]
    y = y1[i] - y2[i]
    z = z1[i] - z2[i]

    a = calculate_angle(x, y, z, 0, 0, 1)
    a_list.append(a)


plt.plot(a_list)
plt.show()
    


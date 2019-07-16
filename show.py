import numpy as np 
import matplotlib.pyplot as plt 

import sys

path = sys.argv[1]
num = int(sys.argv[2])
zzz=np.load(path,allow_pickle=True).item()

cam=zzz['cam'][num]
plt.imshow(cam)
plt.show()
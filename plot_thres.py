import numpy as np 
import matplotlib.pyplot as plt 

import sys
import pickle

class_name = ['background','aeroplane', 'bicycle', 'bird', 'boat',
				'bottle', 'bus', 'car', 'cat', 'chair',
				'cow', 'diningtable', 'dog', 'horse',
				'motorbike', 'person', 'pottedplant',
				'sheep', 'sofa', 'train',
                'tvmonitor']
                
with open('stats.pkl','rb') as f:
    stats = pickle.load(f)

stats_by_class = {i:{'value':[], 'area':[], 'prod':[], 'iou':[]} for i in range(1,21)} #'value':[], 'area':[],
for k,v in stats.items():
    for c,val in v.items():
        stats_by_class[c]['value'].append(val['value']/20000.)
        stats_by_class[c]['area'].append(val['area']/80000.)
        # stats_by_class[c]['prod'].append(np.power(val['value'],3)/val['area']/100000000.)
        stats_by_class[c]['iou'].append(val['iou'])

for c,v in stats_by_class.items():
    plt.figure()
    plt.plot(v['iou'],label='iou')
    val = np.array(v['value'])
    area = np.array(v['area'])
    # import pdb;pdb.set_trace()
    val_diff = val[:-1] - val[1:]
    val_diff = np.append(0,val_diff)
    area_diff = area[:-1] - area[1:]
    area_diff = np.append(1,area_diff)
    speed = val_diff/area_diff
    accl = speed[1:]-speed[:-1]
    accl = np.append(0,accl)
    plt.plot(accl,label='accl')
    plt.legend()
    plt.title('class '+class_name[c])
    plt.savefig('plot_'+str(c)+'.png')
    plt.close()
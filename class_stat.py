import pickle

with open('clss_stat.pkl','rb') as f:
    stat = pickle.load(f)

clss_cnt = stat['class_cnt']
cocur_cnt = stat['cocur_cnt']

class_name = ['aeroplane', 'bicycle', 'bird', 'boat',
				'bottle', 'bus', 'car', 'cat', 'chair',
				'cow', 'diningtable', 'dog', 'horse',
				'motorbike', 'person', 'pottedplant',
				'sheep', 'sofa', 'train',
				'tvmonitor']

clss_cnt_dict = {class_name[i]:clss_cnt[i] for i in range(20)}
print(clss_cnt_dict)

# top 20 cocur:
print('- Top 20 co-occur class pairs:')
import numpy as np
import bottleneck as bn

def top_n_indexes(arr, n):
    idx = bn.argpartition(arr, arr.size-n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]
topn = 35
cocur_cnt = cocur_cnt/(clss_cnt[:,None]*clss_cnt[None,:])
top20_inds = top_n_indexes(cocur_cnt, topn)

top20_vals = np.zeros(topn)
for i in range(topn):
    ind = top20_inds[i]
    top20_vals[i] = cocur_cnt[ind[0],ind[1]]
sortinds = np.argsort(top20_vals)[::-1]
# import pdb;pdb.set_trace()
# top20_vals = top20_vals[sortinds]
# top20_inds = top20_inds[sortinds]

for i in sortinds:
    ind = top20_inds[i]
    print(class_name[ind[0]]+', '+class_name[ind[1]]+': '+str(np.round(cocur_cnt[ind[0],ind[1]]*10000,2)),end=';\t')
print('')


pass
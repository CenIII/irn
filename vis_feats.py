import pickle
import numpy as np 
import time
# import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class_name = ['aeroplane', 'bicycle', 'bird', 'boat',
	'bottle', 'bus', 'car', 'cat', 'chair',
	'cow', 'diningtable', 'dog', 'horse',
	'motorbike', 'person', 'pottedplant',
	'sheep', 'sofa', 'train',
	'tvmonitor']
cdict = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple', 5: 'orange', 6: 'brown', 7: 'pink', 8: 'gray',
			9: 'olive', 10: 'cyan', 11: 'lime', 12: 'deepskyblue', 13: 'yellow', 14: 'gold', 15: 'peru', 
			16: 'teal', 17: 'midnightblue', 18: 'blueviolet', 19: 'tan', 20: 'silver', 21: 'black'} #{i:color_list[i] for i in range(1,21)}

with open('featDict.pkl','rb') as f:
	featDict = pickle.load(f)

# plot discriminative region
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, init='pca', n_iter=500)
# x = np.array(featDict['dis_ft'])
# x = (x / np.linalg.norm(x,axis=1)[:,None])*10
# tsne_results = tsne.fit_transform(x) 

# plt.figure(figsize=(16,20))

# scatter_x = tsne_results[:,0]
# scatter_y = tsne_results[:,1]
# group = featDict['class_id']

# fig, ax = plt.subplots()
# for g in np.unique(group):
# 	ix = np.where(group == g)
# 	ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = class_name[g-1], s = 2)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('feats_vis.png',bbox_inches='tight')
# plt.close()

# use params obtained above, check undiscrm region's distributions. 
tsne = TSNE(n_components=2, verbose=1, perplexity=40, init='pca', n_iter=500)
x = np.array(featDict['dis_ft']+featDict['undis_ft'])
x = (x / np.linalg.norm(x,axis=1)[:,None])*10
tsne_results = tsne.fit_transform(x) 

plt.figure(figsize=(16,20))

scatter_x = tsne_results[:,0]
scatter_y = tsne_results[:,1]
group = featDict['class_id']

fig, ax = plt.subplots()
cnt=1
for g in np.unique(group):
	if cnt>3:
		break
	ix = np.where(group == g)
	ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = class_name[g-1], s = 2)
	cnt+=1
cnt=1
for g in np.unique(group):
	if cnt>3:
		break
	ix = np.where(group == g)[0]+len(group)
	ax.scatter(scatter_x[ix], scatter_y[ix], s = 12, marker='s',edgecolor=cdict[g], linewidth=1, facecolor='none') #label = class_name[g-1],
	cnt+=1
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('feats_vis_undis.png',bbox_inches='tight')

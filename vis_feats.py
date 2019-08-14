import pickle
import numpy as np 
import time
# import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
import matplotlib.colors as mcolors

with open('featDict.pkl','rb') as f:
	featDict = pickle.load(f)

class_name = ['aeroplane', 'bicycle', 'bird', 'boat',
	'bottle', 'bus', 'car', 'cat', 'chair',
	'cow', 'diningtable', 'dog', 'horse',
	'motorbike', 'person', 'pottedplant',
	'sheep', 'sofa', 'train',
	'tvmonitor']
# pca_50 = PCA(n_components=50)
# pca_result_50 = pca_50.fit_transform(featDict['dis_ft'])
# print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

tsne = TSNE(n_components=2, verbose=1, perplexity=40, init='pca', n_iter=500) #perplexity=40,
tsne_results = tsne.fit_transform(np.array(featDict['dis_ft'])) #np.array(featDict['dis_ft'])#
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# df_subset['tsne-2d-one'] = tsne_results[:,0]
# df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,20))
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=df_subset,
#     legend="full",
#     alpha=0.3
# )
scatter_x = tsne_results[:,0]
scatter_y = tsne_results[:,1]
group = featDict['class_id']
color_list = list(mcolors.CSS4_COLORS.keys())
cdict = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple', 5: 'orange', 6: 'brown', 7: 'pink', 8: 'gray',
			9: 'olive', 10: 'cyan', 11: 'lime', 12: 'deepskyblue', 13: 'yellow', 14: 'gold', 15: 'peru', 
			16: 'teal', 17: 'midnightblue', 18: 'blueviolet', 19: 'tan', 20: 'silver', 21: 'black'} #{i:color_list[i] for i in range(1,21)}

fig, ax = plt.subplots()
cnt = 0
for g in np.unique(group):
	# if cnt>3:
	# 	break
	ix = np.where(group == g)
	ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = class_name[g-1], s = 2)
	cnt+=1
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()
plt.savefig('feats_vis.png',bbox_inches='tight')
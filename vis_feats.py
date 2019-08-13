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

with open('featDict_cam_norm.pkl','rb') as f:
	featDict = pickle.load(f)


# pca_50 = PCA(n_components=50)
# pca_result_50 = pca_50.fit_transform(featDict['dis_ft'])
# print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

tsne = TSNE(n_components=2, verbose=1, perplexity=30, init='pca', n_iter=300) #perplexity=40,
tsne_results = tsne.fit_transform(np.array(featDict['dis_ft'])*10)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# df_subset['tsne-2d-one'] = tsne_results[:,0]
# df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
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
cdict = {i:color_list[i] for i in range(1,21)}#{1: 'red', 2: 'blue', 3: 'green'}

fig, ax = plt.subplots()
cnt = 0
for g in np.unique(group):
	# if cnt>2:
	# 	break
	ix = np.where(group == g)
	ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 2)
	cnt+=1
ax.legend()
# plt.show()
plt.savefig('feats_vis.png')
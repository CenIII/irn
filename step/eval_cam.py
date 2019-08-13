
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio
from misc import torchutils, imutils
import tqdm
import torch
import numpy as np
import importlib
import cv2
import pickle
import torch.nn.functional as F

def get_dis_ft(feats,pred,keys):
	D,H,W = feats.shape
	mean_ft_list = []
	var_ft_list = []
	feats_flat = np.reshape(feats,(D,-1))
	for k in keys:
		mask = (pred==k).astype(np.int)
		mask = cv2.resize(mask,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
		mask_flat = np.reshape(mask,(-1))
		nz_indices = np.nonzero(mask_flat)[0]
		if len(nz_indices)==0:
			return None, None
		nz_feats = feats_flat[:,nz_indices]
		mean_ft = nz_feats.mean(axis=1)
		mean_ft_list.append(mean_ft)
		var_ft = nz_feats.var(axis=1)
		var_ft_list.append(var_ft)
	return mean_ft_list, var_ft_list

def get_undis_ft(feats, pred, label, keys):
	D,H,W = feats.shape
	mean_ft_list = []
	var_ft_list = []
	feats_flat = np.reshape(feats,(D,-1))
	for k in keys:
		mask = (pred==k).astype(np.int)
		mask = cv2.resize(mask,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
		mask_flat = np.reshape(mask,(-1))
		mask_l = (label==k).astype(np.int)
		mask_l = cv2.resize(mask_l,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
		mask_l_flat = np.reshape(mask_l,(-1))
		tmp = mask_l_flat - mask_flat
		tmp[tmp<=0] = 0
		nz_indices = np.nonzero(tmp)[0] # FIXME: what if empty? 
		if len(nz_indices)==0:
			return None, None
		nz_feats = feats_flat[:,nz_indices]
		mean_ft = nz_feats.mean(axis=1)
		mean_ft_list.append(mean_ft)
		var_ft = nz_feats.var(axis=1)
		var_ft_list.append(var_ft)
	return mean_ft_list, var_ft_list

def get_bg_ft(feats, label):
	return get_dis_ft(feats, label, [0])

def run(args):
	model = getattr(importlib.import_module(args.cam_network), 'VIS')()
	model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
	model.eval()
	model = model.cuda()

	dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
	labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
	
	preds = []
	featDict={'dis_ft':[],'undis_ft':[],'bg_ft':[],'dis_var':[], 'undis_var':[],'bg_var':[],'class_id':[]}
	# for id in dataset.ids:
	qdar = tqdm.tqdm(range(len(dataset.ids)), total=len(dataset.ids), ascii=True)
	for i in qdar:
		id = dataset.ids[i]
		# get image
		img = torch.from_numpy(dataset._get_image(i)).cuda().unsqueeze(0)
		# pass forward and get feats
		feats = model(img).squeeze()#.data.cpu().numpy()
		feats = F.normalize(feats,dim=0).data.cpu().numpy()*10
		# TODO: get cam from original results
		cam_dict = np.load(os.path.join('./exp/original_cam/result/cam/', id + '.npy'), allow_pickle=True).item()
		cams = cam_dict['high_res']
		cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
		keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
		cls_labels = np.argmax(cams, axis=0)
		if args.cam_eval_use_crf:
			img = np.asarray(imageio.imread(args.voc12_root+"JPEGImages/"+str(id)+'.jpg')) # load the original image 
			pred = imutils.crf_inference_label(img, cls_labels, n_labels=keys.shape[0]) # pass through CRF
			cls_labels = keys[pred]
		else:
			cls_labels = keys[cls_labels]
		
		# for this image, get 1. discri mean feat vec, 2. un-discri mean feat vec, variance, 3. back mean feat vec., variance
		
		mean_dis_ft, var_dis_ft = get_dis_ft(feats,cls_labels,keys[1:])  # two lists
		mean_undis_ft, var_undis_ft = get_undis_ft(feats,labels[i],cls_labels,keys[1:])  # two lists
		if not (mean_dis_ft is None or mean_undis_ft is None):
			featDict['dis_ft'] += mean_dis_ft
			featDict['dis_var'] += var_dis_ft
			featDict['undis_ft'] += mean_undis_ft
			featDict['undis_var'] += var_undis_ft
			featDict['class_id'] += list(keys[1:])
		mean_ft, var_ft = get_bg_ft(feats,labels[i])  # two lists
		if mean_ft is not None:
			featDict['bg_ft'] += mean_ft
			featDict['bg_var'] += var_ft
  
		preds.append(cls_labels.copy())

	with open('featDict_cam_norm.pkl', 'wb') as f:
		pickle.dump(featDict, f)

	confusion = calc_semantic_segmentation_confusion(preds, labels)

	gtj = confusion.sum(axis=1)
	resj = confusion.sum(axis=0)
	gtjresj = np.diag(confusion)
	denominator = gtj + resj - gtjresj
	iou = gtjresj / denominator

	print({'iou': iou, 'miou': np.nanmean(iou)})

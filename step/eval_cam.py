
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
from voc12.dataloader import TorchvisionNormalize
from misc import imutils

# def get_dis_ft(feats,pred,keys):
# 	D,H,W = feats.shape
# 	mean_ft_list = []
# 	var_ft_list = []
# 	feats_flat = np.reshape(feats,(D,-1))
# 	for k in keys:
# 		mask = (pred==k).astype(np.int)
# 		mask = cv2.resize(mask,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
# 		mask_flat = np.reshape(mask,(-1))
# 		nz_indices = np.nonzero(mask_flat)[0]
# 		if len(nz_indices)==0:
# 			return None, None
# 		nz_feats = feats_flat[:,nz_indices]
# 		mean_ft = nz_feats.mean(axis=1)
# 		mean_ft_list.append(mean_ft)
# 		var_ft = nz_feats.var(axis=1)
# 		var_ft_list.append(var_ft)
# 	return mean_ft_list, var_ft_list

def get_dis_ft(feats,cams,keys):
	D,H,W = feats.shape
	mean_ft_list = []
	var_ft_list = []
	feats_flat = np.reshape(feats,(D,-1))
	cnt = 1
	for k in keys:
		mask = cams[cnt]
		mask = cv2.resize(mask,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
		mask_flat = np.reshape(mask,(-1))+1e-5
		mean_ft = (feats_flat*mask_flat[None,:]).sum(axis=1)/mask_flat.sum() 
		mean_ft_list.append(mean_ft)
		var_ft = np.average((feats_flat-mean_ft[:,None])**2, weights=mask_flat, axis=1)
		var_ft_list.append(var_ft)
		cnt += 1
	return mean_ft_list, var_ft_list

# def get_undis_ft(feats, cams, label, keys):
# 	D,H,W = feats.shape
# 	mean_ft_list = []
# 	var_ft_list = []
# 	feats_flat = np.reshape(feats,(D,-1))
# 	cnt = 1
# 	for k in keys:
# 		mask = cams[cnt] #(pred==k).astype(np.int)
# 		mask = cv2.resize(mask,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
# 		mask_flat = np.reshape(mask,(-1))

# 		mask_l = (label==k).astype(np.int)
# 		mask_l = cv2.resize(mask_l,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
# 		mask_l_flat = np.reshape(mask_l,(-1))
		
# 		nz_indices = np.nonzero(mask_l_flat)[0] # FIXME: what if empty? 
# 		if len(nz_indices)==0:
# 			return None, None
# 		nz_feats = feats_flat[:,nz_indices]
# 		weights = ((1-mask_flat)*mask_l_flat)[nz_indices]+1e-5
		
# 		mean_ft = (nz_feats*weights[None,:]).sum(axis=1)/weights.sum()
# 		mean_ft_list.append(mean_ft)
# 		var_ft = np.average((nz_feats-mean_ft[:,None])**2, weights=weights, axis=1)
# 		var_ft_list.append(var_ft)
# 		cnt += 1
# 	return mean_ft_list, var_ft_list

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

def get_diff_ft(feats, pred, pred_rel, label, keys):
	D,H,W = feats.shape
	mean_ft_list = []
	var_ft_list = []
	feats_flat = np.reshape(feats,(D,-1))
	for k in keys:
		mask = (pred==k).astype(np.int)
		mask = cv2.resize(mask,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
		mask_flat = np.reshape(mask,(-1))
		mask_rel = (pred_rel==k).astype(np.int)
		mask_rel = cv2.resize(mask_rel,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
		mask_rel_flat = np.reshape(mask_rel,(-1))

		mask_l = (label==k).astype(np.int)
		mask_l = cv2.resize(mask_l,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
		mask_l_flat = np.reshape(mask_l,(-1))
		tmp = mask_l_flat - mask_flat
		tmp[tmp<=0] = 0
		tmp = (1-tmp)*mask_rel_flat
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
	D,H,W = feats.shape
	mean_ft_list = []
	var_ft_list = []
	feats_flat = np.reshape(feats,(D,-1))
	mask = (label==0).astype(np.int)
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


def run(args):
	model = getattr(importlib.import_module(args.cam_network), 'VIS')()
	model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
	model.eval()
	model = model.cuda()
	img_normal = TorchvisionNormalize()
	dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
	labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
	
	preds = []
	featDict={'dis_ft':[],'undis_ft':[],'bg_ft':[],'dis_var':[], 'undis_var':[], 'diff_ft':[], 'diff_var':[], 'bg_var':[],'class_id':[]}
	# for id in dataset.ids:
	qdar = tqdm.tqdm(range(len(dataset.ids)), total=len(dataset.ids), ascii=True)
	for i in qdar:
		id = dataset.ids[i]
		# get image
		img = dataset._get_image(i)
		img = img_normal(np.moveaxis(img,0,2))
		img = imutils.HWC_to_CHW(img)
		img = torch.from_numpy(img).cuda().unsqueeze(0)
		# pass forward and get feats
		feats = model(img).squeeze()#.data.cpu().numpy()
		feats = feats.data.cpu().numpy() #*10
		# TODO: get cam from original results
		cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item() #'./exp/original_cam/result/cam/'
		cams = cam_dict['high_res']
		cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
		keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
		cls_labels = np.argmax(cams, axis=0)

		cam_dict_rel = np.load(os.path.join('./exp/hier_shwkq_nosqrt01/result/cam', id + '.npy'), allow_pickle=True).item() #'./exp/original_cam/result/cam/'
		cams_rel = cam_dict_rel['high_res']
		cams_rel = np.pad(cams_rel, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
		# keys = np.pad(cam_dict_rel['keys'] + 1, (1, 0), mode='constant')
		cls_labels_rel = np.argmax(cams_rel, axis=0)

		
		if args.cam_eval_use_crf:
			img = np.asarray(imageio.imread(args.voc12_root+"JPEGImages/"+str(id)+'.jpg')) # load the original image 
			pred = imutils.crf_inference_label(img, cls_labels, n_labels=keys.shape[0]) # pass through CRF
			cls_labels = keys[pred]
		else:
			cls_labels = keys[cls_labels]
			cls_labels_rel = keys[cls_labels_rel]
		# for this image, get 1. discri mean feat vec, 2. un-discri mean feat vec, variance, 3. back mean feat vec., variance
		
		mean_dis_ft, var_dis_ft = get_dis_ft(feats,cams,keys[1:])  # two lists
		mean_undis_ft, var_undis_ft = get_undis_ft(feats,cls_labels,labels[i],keys[1:])  # two lists
		mean_diff_ft, var_diff_ft = get_diff_ft(feats,cls_labels, cls_labels_rel, labels[i],keys[1:])
		if not (mean_dis_ft is None or mean_undis_ft is None or mean_diff_ft is None):
			featDict['dis_ft'] += mean_dis_ft
			featDict['dis_var'] += var_dis_ft
			featDict['undis_ft'] += mean_undis_ft
			featDict['undis_var'] += var_undis_ft
			featDict['diff_ft'] += mean_diff_ft
			featDict['diff_var'] += var_diff_ft
			featDict['class_id'] += list(keys[1:])
		mean_ft, var_ft = get_bg_ft(feats,labels[i])  # two lists
		if mean_ft is not None:
			featDict['bg_ft'] += mean_ft
			featDict['bg_var'] += var_ft
  
		preds.append(cls_labels.copy())

	with open('featDict.pkl', 'wb') as f:
		pickle.dump(featDict, f)

	confusion = calc_semantic_segmentation_confusion(preds, labels)

	gtj = confusion.sum(axis=1)
	resj = confusion.sum(axis=0)
	gtjresj = np.diag(confusion)
	denominator = gtj + resj - gtjresj
	iou = gtjresj / denominator

	print({'iou': iou, 'miou': np.nanmean(iou)})

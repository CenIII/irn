import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils, imutils

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

from torch import autograd
import random

from chainercv.datasets import VOCSemanticSegmentationDataset

from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import imageio
import tqdm
from chainercv.evaluations import calc_semantic_segmentation_confusion
import copy 
from PIL import Image
'''
utils
'''
def determine_routine(ep, args):
	# TODO: need to fix.
	routine = 'init'
	# if args.cam_preload:
	# 	ep = ep + 5
	if ep >= 5:
		if ep % 2:
			routine = 'clsbd'
		else:
			routine = 'model'
	return routine

def eval_metrics(split_name, label_dir, args, logger=None):
	# import pdb;pdb.set_trace()
	dataset = VOCSemanticSegmentationDataset(split=split_name, data_dir=args.voc12_root)
	labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
	preds = []
	qdar = tqdm.tqdm(dataset.ids,total=len(dataset.ids),ascii=True)
	for id in qdar:
		cls_labels = imageio.imread(os.path.join(label_dir, id + '.png')).astype(np.uint8)
		# cls_labels[cls_labels > 20] = 0
		preds.append(cls_labels.copy())

	confusion = calc_semantic_segmentation_confusion(preds, labels)#[:21, :21] #[labels[ind] for ind in ind_list]

	gtj = confusion.sum(axis=1)
	resj = confusion.sum(axis=0)
	gtjresj = np.diag(confusion)
	denominator = gtj + resj - gtjresj
	fp = (1. - gtj / denominator)[:21]
	fn = (1. - resj / denominator)[:21]
	iou = (gtjresj / denominator)[:21]

	print("fp and fn:")
	print("fp: "+str(np.round(fp,3)))
	print("fn: "+str(np.round(fn,3)))
	# print(fp[0], fn[0])
	print(np.mean(fp[1:]), np.mean(fn[1:]))
	miou = np.nanmean(iou)
	print({'iou': iou, 'miou': miou})
	if logger is not None:
		logger.write("fp and fn:\n")
		logger.write("fp: "+str(np.round(fp,3))+"\n")
		logger.write("fn: "+str(np.round(fn,3))+"\n")
		logger.write(str(np.mean(fp[1:]))+' '+str(np.mean(fn[1:]))+"\n")
		logger.write(str({'iou': iou, 'miou': miou})+"\n")
		logger.flush()
	return miou
'''
Visualize
'''
def visualize(x, net, hms, label, cb, iterno, img_denorm, savepath):
			# plt.figure(1)
	fig, ax = plt.subplots(nrows=2, ncols=2)
	x = img_denorm(x[0].permute(1,2,0).data.cpu().numpy()).astype(np.int32)
	hm = net.getHeatmaps(hms, label.max(dim=1)[1])
	# plot here
	img = ax[0][0].imshow(x)
	for i in range(1, len(hms)+1):

		img = ax[int(i/2)][int(i%2)].imshow(hm[i-1][0].data.cpu().numpy())
		if cb[i] is not None:
			cb[i].remove()

		divider = make_axes_locatable(ax[int(i/2)][int(i%2)])
		cax = divider.append_axes("right", size="5%", pad=0.05)
		cb[i] = plt.colorbar(img, cax=cax)

	fig.suptitle('iteration '+str(iterno))

	# plt.pause(0.02)
	plt.savefig(os.path.join(savepath, 'visual_train_'+str(iterno)+'.png'))
	plt.close()
	return cb

def visualize_all_classes(hms, label, iterno, savepath, origin=0, descr='orig'):
	# plt.figure(2)
	class_name = ['BG','aeroplane', 'bicycle', 'bird', 'boat',
				'bottle', 'bus', 'car', 'cat', 'chair',
				'cow', 'diningtable', 'dog', 'horse',
				'motorbike', 'person', 'pottedplant',
				'sheep', 'sofa', 'train',
				'tvmonitor']

	fig, ax = plt.subplots(nrows=5, ncols=5)
	hms = hms[origin]# if origin else hms[-1]
	N,W,H,C = hms.shape
	for i in range(0, C):
		ax[int(i/5)][int(i%5)].imshow(hms[0][...,i].data.cpu().numpy())
		peak_val = hms[0][...,i].view(-1).max().data.cpu().numpy()
		ax[int(i/5)][int(i%5)].set_title(class_name[i]+': '+str(np.round(peak_val,2)),color='r' if label[0][i]>0 else 'black')
	fig.suptitle('iteration '+str(iterno))

	# plt.pause(0.02)
	savename = 'visual_train_'+str(iterno)+'_'+descr+'.png'

	plt.savefig(os.path.join(savepath, savename))
	plt.close()
'''
Init train
'''
def model_init_train(train_data_loader, val_data_loader, model, optimizer, avg_meter, timer, args):
	model.train()
	for step, pack in enumerate(train_data_loader):

		img = pack['img'].cuda()
		label = pack['label'].cuda(non_blocking=True)

		x = model(img)
		# TODO: x is has not been gap yet.
		# TODO: apply mask to loss calculation
		loss = F.multilabel_soft_margin_loss(x, label)

		avg_meter.add({'loss1': loss.item()})
		with autograd.detect_anomaly():
			optimizer.zero_grad()
			loss.backward()
			clip_grad_norm_(model.parameters(),1.)
			optimizer.step()

		if (optimizer.global_step-1)%100 == 0:
			timer.update_progress(optimizer.global_step / max_step)

			print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
					'loss:%.4f' % (avg_meter.pop('loss1')),
					'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
					'lr: %.4f' % (optimizer.param_groups[0]['lr']),
					'etc:%s' % (timer.str_estimated_complete()), flush=True)
	else:
		# validate(model, val_data_loader)
		timer.reset_stage()
		torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')


'''
Seg train
'''

def make_seg_label(crf_out):
	# import pdb;pdb.set_trace()
	crf_out = F.interpolate(crf_out, (32,32), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	highent = (-crf_out * torch.log(torch.clamp(crf_out,1e-5,1.))).sum(dim=1)/np.log(21)
	highent[highent>0.8] = 1.
	highent[highent<1.] = 0
	# crf_out[crf_out<0.3] = 0
	# crf_out[:,-1] += 1e-5
	label = torch.argmax(crf_out,dim=1)
	label[highent>0.5] = 21
	return label

def compute_seg_loss(crit, seg_out, seg_label):
	seg_out_lg = F.log_softmax(seg_out,dim=1)
	loss = crit(seg_out_lg, seg_label)
	return loss

def resize_labels(labels, size):
	"""
	Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
	Other nearest methods result in misaligned labels.
	-> F.interpolate(labels, shape, mode='nearest')
	-> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
	"""
	new_labels = []
	for label in labels:
		label = label.cpu().float().numpy()
		label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
		new_labels.append(np.asarray(label))
	new_labels = torch.LongTensor(new_labels).cuda()
	return new_labels

def model_alternate_train(train_data_loader, model, scheduler, avg_meter, timer, args, ep, logger):
	# temp modification: make seg label for 10000+ images
	train_data_loader.dataset.label_dir = args.sem_seg_out_dir+str(ep-1)
	model = torch.nn.DataParallel(model).cuda().train()
	model.module.train()
	model.module.base.freeze_bn()
	criterion = nn.CrossEntropyLoss(ignore_index=255)

	# crit = nn.NLLLoss(ignore_index=21)

	for step, pack in enumerate(train_data_loader):

		scheduler.optimizer.zero_grad()

		loss = 0
		for i in range(1):
			img = pack['img'].cuda()  # ([16, 3, 512, 512])#[i*10:(i+1)*10]
			label = pack['label'].cuda(non_blocking=True)  # [16, 21]
			# mask = pack['mask'].cuda(non_blocking=True)  # [16, 21]
			seg_label = pack['seg_label'].cuda(non_blocking=True)  # [16, 21]
			# import pdb;pdb.set_trace()
			# 1. forward pass model
			logits = model(img)

			iter_loss = 0
			for logit in logits:
				# Resize labels for {100%, 75%, 50%, Max} logits
				_, _, H, W = logit.shape
				labels_ = resize_labels(seg_label, size=(H, W))
				iter_loss += criterion(logit, labels_.cuda())

			# Propagate backward (just compute gradients wrt the loss)
			# iter_loss /= 2.
			iter_loss.backward()

			# loss = compute_seg_loss(crit, seg_output, seg_label)
			loss += float(iter_loss)

		avg_meter.add({'loss1': loss})
			
		scheduler.optimizer.step()
		scheduler.step()

		if (scheduler.last_epoch-1)%100 == 0:
			timer.update_progress(scheduler.last_epoch / scheduler.iter_max)
			to_write = 'step:%5d/%5d' % (scheduler.last_epoch - 1, scheduler.iter_max) + \
					' loss:%.4f' % (avg_meter.pop('loss1')) + \
					' imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()) + \
					' lr: %.4f' % (scheduler.optimizer.param_groups[0]['lr']) + \
					' etc:%s' % (timer.str_estimated_complete())
			logger.write(to_write+'\n')
			logger.flush()
			print(to_write, flush=True)
	else:
		timer.reset_stage()
		torch.save(model.module.state_dict(), args.cam_weights_name + '_' + str(ep) + '.pth')
	torch.cuda.empty_cache()
	return model.module, scheduler.is_max_step()

# TODO: modify so _seg_label_infer_worker can use
def make_seg2clsbd_label(img,seg_output,label,args,img_denorm):
	'''
	CRF version
	Given seg_output of shape ([16, 21, 32, 32]), we make label for clsbd network.
	return: clsbd_label
	'''
	# 0. detach off
	seg_output = seg_output.detach()
	# 1. upsample
	w,h = seg_output.shape[-2:]
	seg_output = F.interpolate(seg_output, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	# if mask is not None:
	# 	mask = mask.detach()
	# 	mask = F.interpolate(mask, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]

	norm_seg = seg_output / F.adaptive_max_pool2d(seg_output, (1, 1)) + 1e-5
	norm_seg = norm_seg * label[:,:,None,None]
	fg = norm_seg[:,1:]
	img_down = F.interpolate(img, (w*4,h*4), mode='bilinear', align_corners=False)
	N = img_down.shape[0]
	# crf fg_conf
	# crf bg_conf
	fg_conf_cam = F.pad(fg, (0, 0, 0, 0, 1, 0, 0, 0), mode='constant',value=args.conf_fg_thres)
	bg_conf_cam = F.pad(fg, (0, 0, 0, 0, 1, 0, 0, 0), mode='constant',value=args.conf_bg_thres)
	for i in range(N):
		img_dn = img_denorm(img_down[i].permute(1,2,0).data.cpu().numpy()).astype(np.ubyte)
		keys = label[i].nonzero()[:,0].cpu().numpy()
		fg_conf_pane = torch.argmax(fg_conf_cam[i][keys], dim=0).cpu().numpy()
		fg_conf = torch.from_numpy(keys[imutils.crf_inference_label(img_dn, fg_conf_pane, n_labels=keys.shape[0])]).cuda().unsqueeze(0)

		bg_conf_pane = torch.argmax(bg_conf_cam[i][keys], dim=0).cpu().numpy()
		bg_conf = torch.from_numpy(keys[imutils.crf_inference_label(img_dn, bg_conf_pane, n_labels=keys.shape[0])]).cuda().unsqueeze(0)
		
	# combine two confs.
	conf = fg_conf.clone()
	conf[fg_conf == 0] = 255
	conf[bg_conf + fg_conf == 0] = 0
	return conf

# def _seg_label_infer_worker(process_id, model, dataset, args, label_out_dir):
# 	databin = dataset[process_id]
# 	n_gpus = torch.cuda.device_count()
# 	data_loader = DataLoader(databin, batch_size=1, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
# 	img_denorm = torchutils.ImageDenorm()

# 	with torch.no_grad(), cuda.device(process_id+1):
# 		model.cuda()
# 		qdar = tqdm.tqdm(enumerate(data_loader),total=len(data_loader),ascii=True,position=process_id)
# 		for iter, pack in qdar:
# 			img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
# 			img = pack['orig_img'].cuda(non_blocking=True)
# 			orig_img_size = np.asarray(pack['size'])
# 			label = pack['label'].cuda(non_blocking=True)
# 			for k in range(len(pack['img'])):
# 				pack['img'][k] = pack['img'][k].cuda(non_blocking=True)
			
# 			seg_output = model.base.forwardMSF(pack['img']) #(orig_img)#
# 			seg_label = make_seg2clsbd_label(img,seg_output,label,args,img_denorm)
			
# 			# rw_up = F.interpolate(seg_label, scale_factor=16, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
# 			rw_pred = imutils.pil_resize(seg_label[0].cpu().numpy().astype('uint8'),orig_img_size,0)
# 			# rw_pred = torch.argmax(rw_up, dim=0).cpu().numpy()
# 			imageio.imsave(os.path.join(label_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
# 			# imageio.imsave(os.path.join(args.valid_model_out_dir, img_name + '_light.png'), (rw_pred*15).astype(np.uint8))
# 			# imageio.imsave(os.path.join(args.valid_clsbd_out_dir, img_name + '_clsbd.png'), (255*hms[-1][0,...,0].cpu().numpy()).astype(np.uint8))

def _seg_validate_infer_worker(process_id, model, dataset, args, use_crf=True):
	databin = dataset[process_id]
	n_gpus = torch.cuda.device_count()
	data_loader = DataLoader(databin, batch_size=1, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

	with torch.no_grad(), cuda.device(process_id):
		model.cuda()
		qdar = tqdm.tqdm(enumerate(data_loader),total=len(data_loader),ascii=True,position=process_id)
		for iter, pack in qdar:
			img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
			orig_img_size = np.asarray(pack['size'])
			label = pack['label'].cuda(non_blocking=True)
			for k in range(len(pack['img'])):
				pack['img'][k] = pack['img'][k].cuda(non_blocking=True)
			seg_output = model.base.forwardMSF(pack['img']) #(orig_img)#
			rw_up = F.interpolate(seg_output, scale_factor=8, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
			rw_up = F.softmax(rw_up,dim=0)
			rw_pred = torch.argmax(rw_up, dim=0)
			
			# rw_up[rw_up<0.9] = 0
			# rw_mask = rw_up.sum(dim=0)
			# rw_pred[rw_mask==0] = 25

			rw_pred = rw_pred.cpu().numpy()
			# import pdb;pdb.set_trace()
			if use_crf:
				img = np.asarray(imageio.imread(voc12.dataloader.get_img_path(img_name, args.voc12_root)))
				rw_pred = imutils.crf_inference_label(img, rw_pred, n_labels=21)
			imageio.imsave(os.path.join(args.valid_model_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
			# imageio.imsave(os.path.join(args.valid_model_out_dir, img_name + '_light.png'), (rw_pred*10).astype(np.uint8))
			# imageio.imsave(os.path.join(args.valid_clsbd_out_dir, img_name + '_clsbd.png'), (255*hms[-1][0,...,0].cpu().numpy()).astype(np.uint8))
def _seg_validate_infer_worker_on_train(process_id, model, dataset, args, use_crf=True):
	databin = dataset[process_id]
	n_gpus = torch.cuda.device_count()
	data_loader = DataLoader(databin, batch_size=1, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

	with torch.no_grad(), cuda.device(process_id):
		model.cuda()
		qdar = tqdm.tqdm(enumerate(data_loader),total=len(data_loader),ascii=True,position=process_id)
		for iter, pack in qdar:
			img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
			orig_img_size = np.asarray(pack['size'])
			label = pack['label'].cuda(non_blocking=True)
			for k in range(len(pack['img'])):
				pack['img'][k] = pack['img'][k].cuda(non_blocking=True)
			seg_output = model.base.forwardMSF(pack['img']) #(orig_img)#
			rw_up = F.interpolate(seg_output, scale_factor=8, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
			rw_up *= label[0,:,None,None]#*100
			rw_up = F.softmax(rw_up,dim=0)
			rw_pred = torch.argmax(rw_up, dim=0)
			
			keys=label[0].nonzero()[:,0].cpu().numpy()
			dekeys = np.zeros(21)#torch.zeros(22).cuda()
			cnt = 0
			for i in keys:
				dekeys[i] = cnt
				cnt += 1
			# import pdb;pdb.set_trace()
			rw_pred = rw_pred.cpu().numpy()
			rw_pred_crf = dekeys[rw_pred].astype(np.uint8)
			# import pdb;pdb.set_trace()
			if use_crf:
				img = np.asarray(imageio.imread(voc12.dataloader.get_img_path(img_name, args.voc12_root)))
				rw_pred = imutils.crf_inference_label(img, rw_pred_crf, n_labels=len(keys))
				rw_pred = keys[rw_pred]

			# rw_up[rw_up<0.9] = 0
			# rw_mask = rw_up.sum(dim=0)
			# rw_pred[rw_mask==0] = 25

			# rw_pred = rw_pred.cpu().numpy()
			# # import pdb;pdb.set_trace()
			# if use_crf:
			# 	img = np.asarray(imageio.imread(voc12.dataloader.get_img_path(img_name, args.voc12_root)))
			# 	rw_pred = imutils.crf_inference_label(img, rw_pred, n_labels=21)
			imageio.imsave(os.path.join(args.valid_model_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
			imageio.imsave(os.path.join(args.valid_model_out_dir, img_name + '_light.png'), (rw_pred*10).astype(np.uint8))
			# imageio.imsave(os.path.join(args.valid_clsbd_out_dir, img_name + '_clsbd.png'), (255*hms[-1][0,...,0].cpu().numpy()).astype(np.uint8))

def _seg_label_infer_worker(process_id, model, dataset, args, label_out_dir, use_crf=True):
	databin = dataset[process_id]
	n_gpus = torch.cuda.device_count()
	data_loader = DataLoader(databin, batch_size=1, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

	with torch.no_grad(), cuda.device(process_id):
		model.cuda()
		qdar = tqdm.tqdm(enumerate(data_loader),total=len(data_loader),ascii=True,position=process_id)
		for iter, pack in qdar:
			img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
			orig_img_size = np.asarray(pack['size'])
			label = pack['label'].cuda(non_blocking=True)
			for k in range(len(pack['img'])):
				pack['img'][k] = pack['img'][k].cuda(non_blocking=True)
			seg_output = model.base.forwardMSF(pack['img']) #(orig_img)#
			rw_up = F.interpolate(seg_output, scale_factor=8, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
			
			# rw_up_bg = F.softmax(rw_up,dim=0)
			# import pdb;pdb.set_trace()
			# rw_up /= rw_up.contiguous().view(21,-1).max(dim=1)[0][:,None,None]
			rw_up *= label[0,:,None,None]#*100
			rw_up = F.softmax(rw_up,dim=0)
			rw_pred = torch.argmax(rw_up, dim=0)
			
			# rw_up[rw_up<0.7] = 0
			# import pdb;pdb.set_trace()
			fg = rw_up[1:]#;fg[fg<0.99] = 0
			fgmx = fg.view(20,-1).max(dim=1)[0]
			fg[fg<fgmx[:,None,None]*0.9]=0

			bg = rw_up[:1]#;bg[bg<0.99] = 0
			bgmx = bg.view(1,-1).max(dim=1)[0]
			bg[bg<bgmx[:,None,None]*0.999]=0
			rw_up = torch.cat((bg,fg),dim=0)
			rw_up *= label[0,:,None,None]
			rw_mask = rw_up.sum(dim=0)
			rw_pred[rw_mask==0] = 21
			zzz=label[0].nonzero()[:,0]
			keys = torch.cat((zzz,torch.tensor([21]).cuda()),dim=0).cpu().numpy()
			dekeys = np.zeros(22)#torch.zeros(22).cuda()
			cnt = 0
			for i in keys:
				dekeys[i] = cnt
				cnt += 1
			# import pdb;pdb.set_trace()
			rw_pred = rw_pred.cpu().numpy()
			rw_pred_crf = dekeys[rw_pred].astype(np.uint8)
			# import pdb;pdb.set_trace()
			if use_crf:
				img = np.asarray(imageio.imread(voc12.dataloader.get_img_path(img_name, args.voc12_root)))
				rw_pred = imutils.crf_inference_label(img, rw_pred_crf, n_labels=len(keys))
				rw_pred = keys[rw_pred]
			rw_pred[rw_pred==21] = 255
			imageio.imsave(os.path.join(label_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
			# imageio.imsave(os.path.join(args.valid_model_out_dir, img_name + '_light.png'), (rw_pred*10).astype(np.uint8))
			# imageio.imsave(os.path.join(args.valid_clsbd_out_dir, img_name + '_clsbd.png'), (255*hms[-1][0,...,0].cpu().numpy()).astype(np.uint8))


def _seg_validate_infer_worker_double_thres(process_id, model, dataset, args):
	databin = dataset[process_id]
	n_gpus = torch.cuda.device_count()
	data_loader = DataLoader(databin, batch_size=1, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

	with torch.no_grad(), cuda.device(process_id):
		model.cuda()
		qdar = tqdm.tqdm(enumerate(data_loader),total=len(data_loader),ascii=True,position=process_id)
		for iter, pack in qdar:
			img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
			orig_img_size = np.asarray(pack['size'])
			label = pack['label'].cuda(non_blocking=True)
			for k in range(len(pack['img'])):
				pack['img'][k] = pack['img'][k].cuda(non_blocking=True)
			seg_output = model.base.forwardMSF(pack['img']) #(orig_img)#

			# seg_output = F.interpolate(seg_output, scale_factor=8, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
			# rw_pred = torch.argmax(rw_up, dim=0).cpu().numpy()
			norm_seg = seg_output / F.adaptive_max_pool2d(seg_output, (1, 1)) + 1e-5
			norm_seg = norm_seg[:,1:] * label[:,1:,None,None]
			fg = norm_seg
			# crf fg_conf
			# crf bg_conf
			fg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.cam_eval_thres)
			bg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.cam_eval_thres)
			
			max_mask = fg_conf.data.new(fg_conf.shape).fill_(0.)
			fg_conf_pane = torch.argmax(fg_conf, dim=1).unsqueeze(1)
			max_mask = torch.scatter(max_mask,dim=1,index=fg_conf_pane,value=1.)
			fg_conf = fg_conf*max_mask

			max_mask = bg_conf.data.new(bg_conf.shape).fill_(0.)
			bg_conf_pane = torch.argmax(bg_conf, dim=1).unsqueeze(1)
			max_mask = torch.scatter(max_mask,dim=1,index=bg_conf_pane,value=1.)
			bg_conf = bg_conf*max_mask
			bg_conf[:,-1:] = bg_conf[:,-1:]/args.conf_bg_thres#*0.7
			# combine two confs.
			clsbd_label = torch.cat((bg_conf[:,-1:],fg_conf[:,:-1]),dim=1)
			clsbd_label[clsbd_label>0] = 1.
			clsbd_label = F.interpolate(clsbd_label, scale_factor=8, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
			rw_pred = torch.argmax(clsbd_label, dim=0).cpu().numpy()

			imageio.imsave(os.path.join(args.valid_model_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
			imageio.imsave(os.path.join(args.valid_model_out_dir, img_name + '_light.png'), (rw_pred*15).astype(np.uint8))
			# imageio.imsave(os.path.join(args.valid_clsbd_out_dir, img_name + '_clsbd.png'), (255*hms[-1][0,...,0].cpu().numpy()).astype(np.uint8))



def model_validate(model, args, ep, logger, make_label=False):
	# import pdb;pdb.set_trace()
	# 分两步，第一步multiprocess infer结果并保存到validate/model/epoch#
	# 第二步参考eval_cam.py, 从文件夹读取结果并单线程eval结果
	model.eval()
	# step 1: make crf results
	if args.quick_infer:
		dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.quick_list,
																voc12_root=args.voc12_root, scales=args.cam_scales)
		dataset = torchutils.split_dataset(dataset, 1)

		_seg_validate_infer_worker(0, model, dataset, args)

		torch.cuda.empty_cache()
		exit(0)
	else:
		if make_label:
			print('Validate: 1. Making seg label for clsbd...')
			n_gpus = torch.cuda.device_count()
			dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
																	voc12_root=args.voc12_root, scales=args.cam_scales)
			dataset = torchutils.split_dataset(dataset, n_gpus)
			label_out_dir = args.ir_label_out_dir+str(ep)
			os.makedirs(label_out_dir, exist_ok=True)
			multiprocessing.spawn(_seg_label_infer_worker, nprocs=n_gpus, args=(model, dataset, args, label_out_dir), join=True)
			# _seg_label_infer_worker(0, model, dataset, args, label_out_dir)
			# _seg_label_infer_worker(1, model, dataset, args, label_out_dir)
			# _seg_label_infer_worker(2, model, dataset, args, label_out_dir)
			# _seg_label_infer_worker(3, model, dataset, args, label_out_dir)
			miou = eval_metrics('train', label_out_dir, args, logger=logger)
			torch.cuda.empty_cache()
			return
		else:
			print('Validate: 2. Making seg preds for val set...')
			n_gpus = torch.cuda.device_count()
			dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
																	voc12_root=args.voc12_root, scales=args.cam_scales)
			dataset = torchutils.split_dataset(dataset, n_gpus)

			multiprocessing.spawn(_seg_validate_infer_worker_on_train, nprocs=n_gpus, args=(model, dataset, args), join=True)
			# _seg_validate_infer_worker(0,model, dataset, args)
			torch.cuda.empty_cache()
			print('Validate: 3. Eval preds...')
			# step 2: eval results
			miou = eval_metrics('train', args.valid_model_out_dir, args, logger=logger)
			torch.cuda.empty_cache()

			# print('Validate: 2. Making seg preds for val set...using double thres...')
			# n_gpus = torch.cuda.device_count()
			# dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.val_list,
			# 														voc12_root=args.voc12_root, scales=args.cam_scales)
			# dataset = torchutils.split_dataset(dataset, n_gpus)

			# multiprocessing.spawn(_seg_validate_infer_worker_double_thres, nprocs=n_gpus, args=(model, dataset, args), join=True)
			# # _seg_validate_infer_worker(0,model, dataset, args)
			# torch.cuda.empty_cache()
			# print('Validate: 3. Eval preds...')
			# # step 2: eval results
			# miou = eval_metrics('val', args.valid_model_out_dir, args, logger=logger)
			# torch.cuda.empty_cache()
			return miou

'''
Clsbd train
'''

def compute_clsbd_loss(pred):
	# TODO: make sure mask is not needed here.
	pos, neg, pos_fg_sum, pos_bg_sum, neg_sum = pred
	loss = (pos[:,0:1]/pos_bg_sum.sum()).sum()/4.+(pos[:,1:]/pos_fg_sum.sum()).sum()/4.+(neg/neg_sum.sum()).sum()/2.
	return loss

def make_seg_unary(seg_output,label,args, mask=None, orig_size=None):
	# 0. detach off
	seg_output = seg_output.detach()
	# 1. upsample
	w,h = seg_output.shape[-2:]
	strided_size = (w*2,h*2)
	if orig_size is not None:
		strided_size = imutils.get_strided_size(orig_size, 4)
	seg_output = F.interpolate(seg_output, strided_size, mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	if mask is not None:
		mask = mask.detach()
		mask = F.interpolate(mask, strided_size, mode='bilinear', align_corners=False) #[16, 21, 128, 128]

	seg_output = seg_output * label[:,:,None,None]
	norm_seg = F.softmax(seg_output,dim=1) #seg_output / F.adaptive_max_pool2d(seg_output, (1, 1)) + 1e-5
	fg = norm_seg
	
	# fg[fg<0.9] = 0
	# crf fg_conf
	# crf bg_conf

	# fg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.unary_fg_thres)
	# bg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.unary_bg_thres)
	
	# max_mask = fg_conf.data.new(fg_conf.shape).fill_(0.)
	# fg_conf_pane = torch.argmax(fg_conf, dim=1).unsqueeze(1)
	# max_mask = torch.scatter(max_mask,dim=1,index=fg_conf_pane,value=1.)
	# fg_conf = fg_conf*max_mask

	# max_mask = bg_conf.data.new(bg_conf.shape).fill_(0.)
	# bg_conf_pane = torch.argmax(bg_conf, dim=1).unsqueeze(1)
	# max_mask = torch.scatter(max_mask,dim=1,index=bg_conf_pane,value=1.)
	# bg_conf = bg_conf*max_mask
	# bg_conf[:,-1:] = bg_conf[:,-1:]/args.conf_bg_thres#*0.7
	# # combine two confs.
	# clsbd_label = torch.cat((bg_conf[:,-1:],fg_conf[:,:-1]),dim=1)
	# fg[fg<0.1] = 0
	# clsbd_label[clsbd_label>0] = 1.
	# fg[fg>0] = 1
	return fg, mask

def make_seg_unary_from_file(img_name, orig_size=None):
	# import pdb;pdb.set_trace()
	seg_pred = torch.from_numpy(imageio.imread('exp/deeplabv2_cam21_meansig/result/validate/model/'+img_name+'.png')).type(torch.LongTensor).cuda()

	strided_size = imutils.get_strided_size(orig_size, 4)
	unary = torch.zeros((1,21,*orig_size)).cuda()
	unary = torch.scatter(unary,1,seg_pred[None,None,:,:],1)
	unary = F.interpolate(unary, strided_size, mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	
	return unary

def make_seg_unary_bak(seg_output,label,args, mask=None, orig_size=None):
	# 0. detach off
	seg_output = seg_output.detach()
	# 1. upsample
	w,h = seg_output.shape[-2:]
	strided_size = (w*4,h*4)
	if orig_size is not None:
		strided_size = imutils.get_strided_size(orig_size, 4)
	seg_output = F.interpolate(seg_output, strided_size, mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	if mask is not None:
		mask = mask.detach()
		mask = F.interpolate(mask, strided_size, mode='bilinear', align_corners=False) #[16, 21, 128, 128]

	norm_seg = seg_output / F.adaptive_max_pool2d(seg_output, (1, 1)) + 1e-5
	norm_seg = norm_seg * label[:,1:,None,None]
	fg = norm_seg
	# crf fg_conf
	# crf bg_conf
	fg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.unary_fg_thres)
	bg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.unary_bg_thres)
	
	max_mask = fg_conf.data.new(fg_conf.shape).fill_(0.)
	fg_conf_pane = torch.argmax(fg_conf, dim=1).unsqueeze(1)
	max_mask = torch.scatter(max_mask,dim=1,index=fg_conf_pane,value=1.)
	fg_conf = fg_conf*max_mask

	max_mask = bg_conf.data.new(bg_conf.shape).fill_(0.)
	bg_conf_pane = torch.argmax(bg_conf, dim=1).unsqueeze(1)
	max_mask = torch.scatter(max_mask,dim=1,index=bg_conf_pane,value=1.)
	bg_conf = bg_conf*max_mask
	bg_conf[:,-1:] = bg_conf[:,-1:]/args.conf_bg_thres#*0.7
	# combine two confs.
	clsbd_label = torch.cat((bg_conf[:,-1:],fg_conf[:,:-1]),dim=1)
	clsbd_label[clsbd_label>0] = 1.
	return clsbd_label, mask

def _label_to_tensor(label):
	label[label==255.] = 21
	label = label.unsqueeze(1)
	N,_,W,H = label.shape
	mask = torch.zeros(N,22,W,H).cuda()
	mask = mask.scatter_(1,label.type(torch.cuda.LongTensor),1.)
	unary = mask[:,:-1]
	return unary

def clsbd_alternate_train(train_data_loader, clsbd, optimizer, avg_meter, timer, args, ep):
	train_data_loader.dataset.label_dir = args.ir_label_out_dir+str(ep-1)
	clsbd = torch.nn.DataParallel(clsbd).cuda().train()
	clsbd.module.train()
	cb = [None, None, None, None]
	img_denorm = torchutils.ImageDenorm()
	for step, pack in enumerate(train_data_loader):

		img = pack['img'].cuda()  # ([16, 3, 512, 512])
		label = pack['label'].cuda(non_blocking=True)  # [16, 21]
		seg_label = pack['seg_label'].cuda(non_blocking=True)
		mask = pack['mask'].cuda(non_blocking=True)  # [16, 21]
		loss_pack, hms = clsbd(img, _label_to_tensor(seg_label), mask=mask)
		if (optimizer.global_step-1)%20 == 0 and args.cam_visualize_train:
			visualize(img, clsbd.module, hms, label, cb, optimizer.global_step-1, img_denorm, args.vis_out_dir_clsbd)
			visualize_all_classes(hms, label, optimizer.global_step-1, args.vis_out_dir_clsbd, origin=0, descr='unary')

		loss = compute_clsbd_loss(loss_pack)

		avg_meter.add({'loss1': loss.item()})
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (optimizer.global_step-1)%50 == 0:
			timer.update_progress(optimizer.global_step / optimizer.max_step)

			print('step:%5d/%5d' % (optimizer.global_step - 1, optimizer.max_step),
					'loss:%.4f' % (avg_meter.pop('loss1')),
					'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
					'lr: %.4f' % (optimizer.param_groups[0]['lr']),
					'etc:%s' % (timer.str_estimated_complete()), flush=True)
	else:
		timer.reset_stage()
		torch.save(clsbd.module.state_dict(), args.irn_weights_name + '_' + str(ep) + '.pth')
	return clsbd.module, optimizer.is_max_step()

def _clsbd_label_infer_worker(process_id, model, clsbd, dataset, args, label_out_dir):

	databin = dataset[process_id]
	n_gpus = torch.cuda.device_count()
	data_loader = DataLoader(databin, batch_size=1, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

	with torch.no_grad(), cuda.device(process_id):

		model.cuda()
		clsbd.cuda()
		qdar = tqdm.tqdm(enumerate(data_loader),total=len(data_loader),ascii=True,position=process_id)
		for iter, pack in qdar:
			img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
			orig_img_size = np.asarray(pack['size'])
			orig_img = pack['orig_img'].cuda(non_blocking=True)
			label = pack['label'].cuda(non_blocking=True)
			for k in range(len(pack['img'])):
				pack['img'][k] = pack['img'][k].cuda(non_blocking=True)
			
			seg_output = model.base.forwardMSF(pack['img']) #(orig_img)#
			unary_1, _ = make_seg_unary(seg_output,label,args,orig_size=orig_img_size)
			unary = make_seg_unary_from_file(img_name,orig_size=orig_img_size)
			# pack['img'] for clsbd forward

			# smaller kernel
			# clsbd.usecrf(2)
			# rw, hms = clsbd.forwardMSF(pack['img'],unary,num_iter=120) #(orig_img,unary,num_iter=50)#
			# clsbd.usecrf(1)
			rw, hms = clsbd.forwardMSF(pack['img'],unary_1,num_iter=100) #(orig_img,unary,num_iter=50)#
			# import pdb;pdb.set_trace()

			rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
			unary_up = F.interpolate(unary, scale_factor=4, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
			rw_max = torch.argmax(rw_up,dim=0)
			rw_bit = rw_up.data.new(rw_up.shape).fill_(0)
			rw_bit = torch.scatter(rw_bit,0,rw_max[None,:,:],1)
			unary_max = torch.argmax(unary_up,dim=0)
			unary_bit = unary_up.data.new(unary_up.shape).fill_(0)
			unary_bit = torch.scatter(unary_bit,0,unary_max[None,:,:],1)

			mg_fg = (unary_bit + 0.5*rw_bit)[1:]
			mask = mg_fg.sum(dim=0)
			rw_pred = torch.argmax(mg_fg,dim=0) + 1
			rw_pred[mask==0] = 0


			# rw_pred = torch.argmax(rw_up, dim=0)
			# rw_up[rw_up<0.8] = 0
			# rw_mask = rw_up.sum(dim=0)
			# rw_pred[rw_mask==0] = 25
			rw_pred = rw_pred.cpu().numpy()
			imageio.imsave(os.path.join(label_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
			imageio.imsave(os.path.join(label_out_dir, img_name + '_light.png'), (rw_pred*10).astype(np.uint8))
			# imageio.imsave(os.path.join(label_out_dir, img_name + '_clsbd.png'), (255*hms[-2][0,...,0].cpu().numpy()).astype(np.uint8))

def clsbd_validate(model, clsbd, args, ep):
	# 分两步，第一步multiprocess infer结果并保存到validate/clsbd/epoch#
	# 第二步参考eval_cam.py, 从文件夹读取结果并单线程eval结果
	model.eval()
	clsbd.eval()
	# step 1: make crf results
	if args.quick_infer:
		dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.quick_list,
																voc12_root=args.voc12_root, scales=(1.0,1.5,0.5))#args.cam_scales)
		dataset = torchutils.split_dataset(dataset, 1)

		infer_out_dir = args.valid_clsbd_out_dir
		_clsbd_label_infer_worker(0, model, clsbd, dataset, args, infer_out_dir)

		torch.cuda.empty_cache()
		exit(0)
	else:
		print('Validate: 1. Making crf inference labels...')
		n_gpus = torch.cuda.device_count()
		dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
																voc12_root=args.voc12_root, scales=args.cam_scales)
		dataset = torchutils.split_dataset(dataset, n_gpus)
		label_out_dir = args.sem_seg_out_dir+str(ep)
		os.makedirs(label_out_dir, exist_ok=True)
		multiprocessing.spawn(_clsbd_label_infer_worker, nprocs=n_gpus, args=(model, clsbd, dataset, args, label_out_dir), join=True)

		torch.cuda.empty_cache()
		
		print('Validate: 2. Eval labels...')
		# step 2: eval results
		miou = eval_metrics('train', label_out_dir, args)
		exit(0)
		return None#miou
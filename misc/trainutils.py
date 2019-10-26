import torch
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
	class_name = ['aeroplane', 'bicycle', 'bird', 'boat',
				'bottle', 'bus', 'car', 'cat', 'chair',
				'cow', 'diningtable', 'dog', 'horse',
				'motorbike', 'person', 'pottedplant',
				'sheep', 'sofa', 'train',
				'tvmonitor','BG']

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
def make_seg_unary(seg_output,label,args, mask=None, orig_size=None):
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
	norm_seg = norm_seg * label[:,:,None,None]
	fg = norm_seg[:,:-1]
	# crf fg_conf
	# crf bg_conf
	fg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.conf_fg_thres)
	bg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.conf_bg_thres)
	
	max_mask = fg_conf.data.new(fg_conf.shape).fill_(0.)
	fg_conf_pane = torch.argmax(fg_conf, dim=1).unsqueeze(1)
	max_mask = torch.scatter(max_mask,dim=1,index=fg_conf_pane,value=1.)
	fg_conf = fg_conf*max_mask

	max_mask = bg_conf.data.new(bg_conf.shape).fill_(0.)
	bg_conf_pane = torch.argmax(bg_conf, dim=1).unsqueeze(1)
	max_mask = torch.scatter(max_mask,dim=1,index=bg_conf_pane,value=1.)
	bg_conf = bg_conf*max_mask
	bg_conf[:,-1:] = bg_conf[:,-1:]/args.conf_bg_thres*0.7
	# combine two confs.
	clsbd_label = torch.cat((fg_conf[:,:-1],bg_conf[:,-1:]),dim=1)
	return clsbd_label, mask
	# 1. upsample

	# 2. softmax

	# 3. obtain argmax mask???

	# 4. select target classes

	# 5. thresholding fg, bg???
	# pass

def compute_seg_loss():
	pass

def model_alternate_train(train_data_loader, model, clsbd, optimizer, avg_meter, timer, args):
	model.train()
	clsbd.eval()
	for step, pack in enumerate(train_data_loader):

		img = pack['img'].cuda()
		label = pack['label'].cuda(non_blocking=True)

		seg_output = model(img)
		seg_unary = make_seg_unary(seg_output, label, args) # format: one channel image, each pixel denoted by class num.
		crf_output, hms = clsbd(img, seg_unary)
		loss = compute_seg_loss(seg_output,crf_output)

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
		# TODO: validation.
		timer.reset_stage()
		torch.save(model.module.state_dict(), args.irn_weights_name + '.pth')

'''
Clsbd train
'''
def make_seg2clsbd_label(img,seg_output,label,args,img_denorm,mask=None):
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
	if mask is not None:
		mask = mask.detach()
		mask = F.interpolate(mask, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]

	norm_seg = seg_output / F.adaptive_max_pool2d(seg_output, (1, 1)) + 1e-5
	norm_seg = norm_seg * label[:,:,None,None]
	fg = norm_seg[:,:-1]
	img_down = F.interpolate(img, (w*4,h*4), mode='bilinear', align_corners=False)
	N = img_down.shape[0]
	# crf fg_conf
	# crf bg_conf
	fg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.conf_fg_thres)
	bg_conf = F.pad(fg, (0, 0, 0, 0, 0, 1, 0, 0), mode='constant',value=args.conf_bg_thres)
	for i in range(N):
		img_dn = img_denorm(img_down[i].permute(1,2,0).data.cpu().numpy()).astype(np.ubyte)
		keys = label[i].nonzero()[:,0].cpu().numpy()
		fg_conf_pane = torch.argmax(fg_conf[i][keys], dim=0).cpu().numpy()
		pred = torch.from_numpy(keys[imutils.crf_inference_label(img_dn, fg_conf_pane, n_labels=keys.shape[0])]).cuda().unsqueeze(0)
		fg_conf[i] = 0.
		fg_conf[i] = torch.scatter(fg_conf[i],dim=0,index=pred,value=1.)

		bg_conf_pane = torch.argmax(bg_conf[i][keys], dim=0).cpu().numpy()
		pred = torch.from_numpy(keys[imutils.crf_inference_label(img_dn, bg_conf_pane, n_labels=keys.shape[0])]).cuda().unsqueeze(0)
		bg_conf[i] = 0.
		bg_conf[i] = torch.scatter(bg_conf[i],dim=0,index=pred,value=1.)
	# combine two confs.
	clsbd_label = torch.cat((fg_conf[:,:-1],bg_conf[:,-1:]),dim=1)
	return clsbd_label, mask

def compute_clsbd_loss(pred):
	# TODO: make sure mask is not needed here.
	pos, neg, pos_fg_sum, pos_bg_sum, neg_sum = pred
	loss = (pos[:,-1:]/pos_bg_sum.sum()).sum()/4.+(pos[:,:-1]/pos_fg_sum.sum()).sum()/4.+(neg/neg_sum.sum()).sum()/2.
	return loss

def clsbd_alternate_train(train_data_loader, model, clsbd, optimizer, avg_meter, timer, args):
	model.eval()
	clsbd.train()
	cb = [None, None, None, None]
	img_denorm = torchutils.ImageDenorm()
	for step, pack in enumerate(train_data_loader):

		img = pack['img'].cuda()  # ([16, 3, 512, 512])
		img_pack = [pack['msf_img'][i].cuda() for i in range(len(pack['msf_img']))]
		label = pack['label'].cuda(non_blocking=True)  # [16, 21]
		mask = pack['mask'].cuda(non_blocking=True)  # [16, 21]
		# mask down sample
		seg_output = model(img_pack,MSF=True) # [16, 21, 32, 32]
		seg_label, mask = make_seg2clsbd_label(img, seg_output, label, args, img_denorm, mask=mask) # format: one channel image, each pixel denoted by class num.
		
		loss_pack, hms = clsbd(img, seg_label, mask=mask)
		if (optimizer.global_step-1)%20 == 0 and args.cam_visualize_train:
			visualize(img, clsbd.module, hms, label, cb, optimizer.global_step-1, img_denorm, args.vis_out_dir_clsbd)
			visualize_all_classes(hms, label, optimizer.global_step-1, args.vis_out_dir_clsbd, origin=0, descr='unary')

		loss = compute_clsbd_loss(loss_pack)

		avg_meter.add({'loss1': loss.item()})
		with autograd.detect_anomaly():
			optimizer.zero_grad()
			loss.backward()
			# clip_grad_norm_(clsbd.parameters(),1.)
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
		torch.save(clsbd.module.state_dict(), args.irn_weights_name + '.pth')

def _clsbd_validate_infer_worker(process_id, model, clsbd, dataset, args):

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
			# pack['orig_img'] for model forward to make unary, call "make_seg_unary" here
			# import pdb;pdb.set_trace()
			seg_output = model(orig_img)
			unary, _ = make_seg_unary(seg_output,label,args,orig_size=orig_img_size)
			# pack['img'] for clsbd forward
			rw, _ = clsbd.forwardMSF(pack['img'],unary)
			rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
			rw_up[rw_up<0.5] = 0
			# ambiguous region classified to bg
			rw_up[-1] += 1e-5
			rw_pred = torch.argmax(rw_up, dim=0).cpu().numpy()

			imageio.imsave(os.path.join(args.valid_clsbd_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
			imageio.imsave(os.path.join(args.valid_clsbd_out_dir, img_name + '_light.png'), (rw_pred*15).astype(np.uint8))


def clsbd_validate(model, clsbd, args):
	# 分两步，第一步multiprocess infer结果并保存到validate/clsbd/epoch#
	# 第二步参考eval_cam.py, 从文件夹读取结果并单线程eval结果
	model.eval()
	clsbd.eval()
	print('Validate: 1. Making crf inference labels...')
	# step 1: make crf results
	n_gpus = torch.cuda.device_count()
	dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
															 voc12_root=args.voc12_root, scales=args.cam_scales)
	dataset = torchutils.split_dataset(dataset, n_gpus)

	# print('[ ', end='')
	# multiprocessing.spawn(_clsbd_validate_infer_worker, nprocs=n_gpus, args=(model, clsbd, dataset, args), join=True)
	# print(']')
	_clsbd_validate_infer_worker(0, model, clsbd, dataset, args)

	torch.cuda.empty_cache()

	print('Validate: 2. Eval labels...')
	# step 2: eval results
	dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
	labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
	preds = []
	qdar = tqdm.tqdm(dataset.ids,total=len(dataset.ids),ascii=True)
	for id in qdar:
		cls_labels = imageio.imread(os.path.join(args.valid_clsbd_out_dir, id + '.png')).astype(np.uint8) + 1
		cls_labels[cls_labels == 21] = 0
		preds.append(cls_labels.copy())

	confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21] #[labels[ind] for ind in ind_list]

	gtj = confusion.sum(axis=1)
	resj = confusion.sum(axis=0)
	gtjresj = np.diag(confusion)
	denominator = gtj + resj - gtjresj
	fp = 1. - gtj / denominator
	fn = 1. - resj / denominator
	iou = gtjresj / denominator

	print("fp and fn:")
	print("fp: "+str(np.round(fp,3)))
	print("fn: "+str(np.round(fn,3)))
	# print(fp[0], fn[0])
	print(np.mean(fp[1:]), np.mean(fn[1:]))

	print({'iou': iou, 'miou': np.nanmean(iou)})
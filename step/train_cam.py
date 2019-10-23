
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
from torch.nn.utils import clip_grad_norm_
import random 

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

def validate(model, data_loader):
	print('validating ... ', flush=True, end='')

	val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

	model.eval()

	with torch.no_grad():
		for pack in data_loader:
			img = pack['img']

			label = pack['label'].cuda(non_blocking=True)

			x = model(img)
			loss1 = F.multilabel_soft_margin_loss(x, label)

			val_loss_meter.add({'loss1': loss1.item()})

	model.train()

	print('loss: %.4f' % (val_loss_meter.pop('loss1')))

	return

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

def make_seg_unary(seg_output,label,args):
	# 1. upsample

	# 2. softmax 
	
	# 3. obtain argmax mask???

	# 4. select target classes

	# 5. thresholding fg, bg???
	pass 

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

def make_seg2clsbd_label_bak(seg_output,label,mask,args):
	'''
	Given seg_output of shape ([16, 21, 32, 32]), we make label for clsbd network. 
	return: clsbd_label
	'''
	# 0. detach off
	seg_output = seg_output.detach()
	mask = mask.detach()
	# 1. upsample
	w,h = seg_output.shape[-2:]
	seg_output = F.interpolate(seg_output, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	mask = F.interpolate(mask, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	# 2. softmax 
	probs = F.softmax(seg_output,dim=1)
	# 3. obtain & apply argmax mask
	preds = torch.argmax(probs,dim=1).unsqueeze(1)
	max_mask = probs.data.new(probs.shape).fill_(0.)
	max_mask = torch.scatter(max_mask,dim=1,index=preds,value=1.)
	probs_m = probs * max_mask
	# 4. select target classes
	probs_mtc = probs_m * label[:,:,None,None]
	# 5. thresholding fg, bg
	fg = probs_mtc[:,:-1]
	bg = probs_mtc[:,-1:]
	fg[fg<args.conf_fg_thres] = 0.
	bg[bg<args.conf_bg_thres] = 0.
	clsbd_label = torch.cat((fg,bg),dim=1)
	# 6. binarize 
	clsbd_label[clsbd_label>0] = 1.
	# 7. apply mask
	clsbd_label = clsbd_label * mask
	return clsbd_label, mask

def make_seg2clsbd_label_norm(seg_output,label,mask,args):
	'''
	Given seg_output of shape ([16, 21, 32, 32]), we make label for clsbd network. 
	return: clsbd_label
	'''
	# 0. detach off
	seg_output = seg_output.detach()
	mask = mask.detach()
	# 1. upsample
	w,h = seg_output.shape[-2:]
	seg_output = F.interpolate(seg_output, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	mask = F.interpolate(mask, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	
	norm_seg = seg_output / F.adaptive_max_pool2d(seg_output, (1, 1)) + 1e-5
	norm_seg = norm_seg * label[:,:,None,None]
	fg = norm_seg[:,:-1]
	bg = norm_seg[:,:-1].clone()
	fg[fg<args.conf_fg_thres] = 0.
	fg[fg>=args.conf_fg_thres] = 1.
	bg[bg<args.conf_bg_thres] = 0.
	bg[bg>=args.conf_bg_thres] = 1.
	bg = bg.sum(dim=1,keepdim=True)
	bg[bg>0] = 1
	bg = 1 - bg
	clsbd_label = torch.cat((fg,bg),dim=1)

	return clsbd_label, mask

def make_seg2clsbd_label(img,seg_output,label,mask,args,img_denorm):
	'''
	CRF version
	Given seg_output of shape ([16, 21, 32, 32]), we make label for clsbd network. 
	return: clsbd_label
	'''
	# 0. detach off
	seg_output = seg_output.detach()
	mask = mask.detach()
	# 1. upsample
	w,h = seg_output.shape[-2:]
	seg_output = F.interpolate(seg_output, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
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
		label = pack['label'].cuda(non_blocking=True)  # [16, 21]
		mask = pack['mask'].cuda(non_blocking=True)  # [16, 21]
		# mask down sample 
		seg_output = model(img) # [16, 21, 32, 32]
		seg_label, mask = make_seg2clsbd_label(img, seg_output, label, mask, args, img_denorm) # format: one channel image, each pixel denoted by class num. 
		# import pdb;pdb.set_trace()
		loss_pack, hms = clsbd(img, seg_label, mask=mask)
		# TODO: visualize
		if (optimizer.global_step-1)%2 == 0 and args.cam_visualize_train:
			visualize(img, clsbd.module, hms, label, cb, optimizer.global_step-1, img_denorm, args.vis_out_dir_clsbd)
			visualize_all_classes(hms, label, optimizer.global_step-1, args.vis_out_dir_clsbd, origin=0, descr='unary')

		loss = compute_clsbd_loss(loss_pack)

		avg_meter.add({'loss1': loss.item()})
		with autograd.detect_anomaly():
			optimizer.zero_grad()
			loss.backward()
			# clip_grad_norm_(clsbd.parameters(),1.)
			optimizer.step()

		if (optimizer.global_step-1)%20 == 0:
			timer.update_progress(optimizer.global_step / optimizer.max_step)

			print('step:%5d/%5d' % (optimizer.global_step - 1, optimizer.max_step),
					'loss:%.4f' % (avg_meter.pop('loss1')),
					'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
					'lr: %.4f' % (optimizer.param_groups[0]['lr']),
					'etc:%s' % (timer.str_estimated_complete()), flush=True)
	else:
		# TODO: validation. 
		timer.reset_stage()
		torch.save(clsbd.module.state_dict(), args.irn_weights_name + '.pth')

def run(args):

	model = getattr(importlib.import_module(args.cam_network), 'Net')()
	if args.cam_preload:
		model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)

	clsbd = getattr(importlib.import_module(args.irn_network), 'Net')()
	seed = 25
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	np.random.seed(seed)  # Numpy module.
	random.seed(seed)  # Python random module.
	torch.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	def _init_fn(worker_id):
		np.random.seed(int(seed))

	# model train loader
	train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
																resize_long=(320, 640), hor_flip=True,
																crop_size=512, crop_method="random",rescale=(0.5, 1.5))
	train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
								   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
	
	max_step = (len(train_dataset) // args.cam_batch_size) #* args.cam_num_epoches
	model_max_step = max(0, max_step * ((args.cam_num_epoches-5)//2))
	if not args.cam_preload:
		model_max_step += max_step * 5
	clsbd_max_step = max(0, max_step * ((args.cam_num_epoches-5+1)//2))

	# model optimizer
	model_param_groups = model.trainable_parameters()
	model_optimizer = torchutils.PolyOptimizer([
		{'params': model_param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
		{'params': model_param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
	], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=model_max_step)
	model = torch.nn.DataParallel(model).cuda()

	# clsbd train loader is the same with model loader. 
	# clsbd optimizer 
	clsbd_param_groups = clsbd.trainable_parameters()
	clsbd_optimizer = torchutils.PolyOptimizer([
		{'params': clsbd_param_groups, 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
	], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=clsbd_max_step)
	clsbd = torch.nn.DataParallel(clsbd).cuda()

	avg_meter = pyutils.AverageMeter()
	timer = pyutils.Timer()

	ep_start = 5 if args.cam_preload else 0
	for ep in range(ep_start, args.cam_num_epoches):
		
		# decide what routine to take, model-init, clsbd or model
		print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))
		rt_key = determine_routine(ep,args)
		if rt_key == 'init':
			model_init_train(train_data_loader, model, model_optimizer, avg_meter, timer, args)
		elif rt_key == 'model':
			model_alternate_train(train_data_loader, model, clsbd, model_optimizer, avg_meter, timer, args)
		elif rt_key == 'clsbd':
			clsbd_alternate_train(train_data_loader, model, clsbd, clsbd_optimizer, avg_meter, timer, args)

	torch.cuda.empty_cache()
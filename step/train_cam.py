
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
import copy
from torch import autograd
from torch.nn.utils import clip_grad_norm_
import random 
import time 

from chainercv.datasets import VOCSemanticSegmentationDataset

from misc.trainutils import determine_routine, clsbd_alternate_train, clsbd_validate, model_alternate_train, model_init_train, model_validate
from net import resnet50

def get_st_steps(ep_start, max_step):
	st_steps = [0,0,0]
	if ep_start<=4:
		st_steps[0] = ep_start * max_step
	else:
		alt_eps = ep_start - 1 - 4
		clsbd_eps = (alt_eps+1)//2
		model_eps = alt_eps//2
		st_steps[1] = model_eps*max_step
		st_steps[2] = clsbd_eps*max_step
	return st_steps

def get_model_optimizer(model,args,max_step):
	# model optimizer
	# model_param_groups = model.trainable_parameters()
	# model_optimizer = torchutils.PolyOptimizer([
	# 	{'params': model_param_groups[0], 'lr': args.seg_learning_rate, 'weight_decay': args.seg_weight_decay},
	# 	{'params': model_param_groups[1], 'lr': 10*args.seg_learning_rate, 'weight_decay': args.seg_weight_decay},
	# ], lr=args.seg_learning_rate, weight_decay=args.seg_weight_decay, max_step=model_max_step, set_step=st_steps[1])

	optimizer = torch.optim.SGD(
		# cf lr_mult and decay_mult in train.prototxt
		params=[
			{
				"params": model.base.get_params(key="1x"),
				"lr": args.seg_learning_rate,
				"weight_decay": args.seg_weight_decay,
			},
			{
				"params": model.base.get_params(key="10x"),
				"lr": 10 * args.seg_learning_rate,
				"weight_decay": args.seg_weight_decay,
			},
			{
				"params": model.base.get_params(key="20x"),
				"lr": 20 * args.seg_learning_rate,
				"weight_decay": 0.0,
			},
		],
		momentum=0.9,
	)
	# Learning rate scheduler
	scheduler = torchutils.PolynomialLR(
		optimizer=optimizer,
		step_size=10,
		iter_max=max_step,
		power=0.9,
	)

	return scheduler 

def get_clsbd_optimizer(clsbd,args,clsbd_max_step):
	# clsbd train loader is the same with model loader. 
	# clsbd optimizer 
	clsbd_param_groups = clsbd.trainable_parameters()
	clsbd_optimizer = torchutils.PolyOptimizer([
		{'params': clsbd_param_groups[0], 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
		{'params': clsbd_param_groups[1], 'lr': 0.1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
	], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=clsbd_max_step, set_step=0)

	return clsbd_optimizer 

def reload_res50(model):
	model_dict = model.base.state_dict()
	checkpoint = resnet50.get_resnet50_state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {}
	for k, v in checkpoint.items():
		# if 'module' in k:
		# 	k = k[7:]
		if(k in model_dict):
			pretrained_dict[k] = v
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	model.base.load_state_dict(model_dict)
	return model 

def run(args):

	model_init = getattr(importlib.import_module(args.cam_network), 'Net')()
	if args.cam_preload:
		model_init.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)

	model = getattr(importlib.import_module(args.seg_network), 'DeepLabV2_ResNet50_MSC')(21)

	clsbd = getattr(importlib.import_module(args.irn_network), 'Net')()
	if args.clsbd_preload:
		clsbd.load_state_dict(torch.load(args.irn_weights_name), strict=False)

	# model train loader
	model_train_dataset = voc12.dataloader.VOC12ClassificationDatasetMSF_TrainModel(args.train_list, voc12_root=args.voc12_root, label_dir=args.sem_seg_out_dir, #resize_long=(320, 640),
																hor_flip=True,
																crop_size=512, crop_method="random",resize_long=(320, 640), scales=(1,0.75,0.5, 1.25, 1.5))
	model_train_data_loader = DataLoader(model_train_dataset, batch_size=args.cam_batch_size,
								   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)#, worker_init_fn=_init_fn)

	# clsbd train loader 
	clsbd_train_dataset = voc12.dataloader.VOC12ClassificationDatasetMSF_TrainClsbd(args.train_list, voc12_root=args.voc12_root, label_dir=args.ir_label_out_dir, #resize_long=(320, 640),
																hor_flip=True,
																crop_size=512, crop_method="random",rescale=(0.5, 1.5))
	clsbd_train_data_loader = DataLoader(clsbd_train_dataset, batch_size=args.irn_batch_size,
								   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)#, worker_init_fn=_init_fn)


	max_step = (len(model_train_dataset) // args.cam_batch_size) #* args.cam_num_epoches
	model_init_max_step = 5
	model_max_step = max(0, max_step * ((args.cam_num_epoches-5)//2))
	clsbd_max_step = max(0, max_step * ((args.cam_num_epoches-5+1)//2))
	ep_start = 5 if args.cam_preload else 0
	ep_start = max(args.cam_start_epoch, ep_start)
	st_steps = get_st_steps(ep_start,max_step)

	avg_meter = pyutils.AverageMeter()
	timer = pyutils.Timer()
	logger = open(args.log_file_path,'a')
	logger.write('!!! '+'Exp time: '+time.ctime()+'\n')

	for ep in range(ep_start, args.cam_num_epoches):
		
		# decide what routine to take, model-init, clsbd or model
		print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))
		rt_key = determine_routine(ep,args)
		if rt_key == 'model':
			model = getattr(importlib.import_module(args.seg_network), 'DeepLabV2_ResNet50_MSC')(21)
			# model = reload_res50(model)
			model.load_state_dict(torch.load('exp/deeplabv2_cam21_meansig/sess/res50_cam_8.pth'), strict=False)
			model_optimizer = get_model_optimizer(model, args, 10*max_step)
			best_miou = 0
			miou = -1
			is_max_step = False
			# model_optimizer.last_epoch = 1068
			# model_optimizer.step(epoch=2645)
			while True:
				# model_new, is_max_step = model_alternate_train(model_train_data_loader, model, model_optimizer, avg_meter, timer, args, ep, logger)
				# import pdb;pdb.set_trace()
				miou = model_validate(model, args, ep, logger, make_label=True)
				exit(0)
				if is_max_step: #miou < best_miou or 
					# model_validate(model_new, args, ep, make_label=True)
					exit(0)
					break
				best_miou = miou
				model = model_new
		elif rt_key == 'clsbd':
			# import pdb;pdb.set_trace()
			# model = getattr(importlib.import_module(args.seg_network), 'DeepLabV2_ResNet50_MSC')(21)
			# # # model = reload_res50(model)
			# model.load_state_dict(torch.load('exp/deeplabv2_cam21_meansig/sess/res50_cam_6_637.pth'), strict=False)
			clsbd = getattr(importlib.import_module(args.irn_network), 'Net')()
			clsbd.load_state_dict(torch.load('exp/deeplabv2_cam21_meansig/sess/res50_clsbd_5.pth'), strict=False)

			# clsbd_optimizer = get_clsbd_optimizer(clsbd, args, 3*max_step)
			# is_max_step = False
			# while not is_max_step:
			# 	clsbd, is_max_step = clsbd_alternate_train(clsbd_train_data_loader, clsbd, clsbd_optimizer, avg_meter, timer, args, ep)
			# exit(0)
			# import pdb;pdb.set_trace()
			model_in = model_init if ep==5 else model
			# import pdb;pdb.set_trace()
			clsbd_validate(model_in, clsbd, args, ep)

	torch.cuda.empty_cache()

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

from chainercv.datasets import VOCSemanticSegmentationDataset

from misc.trainutils import determine_routine, clsbd_alternate_train, clsbd_validate, model_alternate_train, model_init_train, model_validate

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

def run(args):

	model = getattr(importlib.import_module(args.cam_network), 'Net')()
	if args.cam_preload:
		model_init = copy.deepcopy(model)
		model_init.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)

	clsbd = getattr(importlib.import_module(args.irn_network), 'Net')()
	if args.clsbd_preload:
		clsbd.load_state_dict(torch.load('exp/original_cam/sess/res50_irn.pth'), strict=False)

	# seed = 25
	# torch.manual_seed(seed)
	# torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	# np.random.seed(seed)  # Numpy module.
	# random.seed(seed)  # Python random module.
	# torch.manual_seed(seed)
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True
	# def _init_fn(worker_id):
	# 	np.random.seed(int(seed))

	# model train loader
	train_dataset = voc12.dataloader.VOC12ClassificationDatasetMSF_Train(args.train_list, voc12_root=args.voc12_root, label_dir=args.sem_seg_out_dir, #resize_long=(320, 640),
																hor_flip=True,
																crop_size=512, crop_method="random",rescale=(1., 1.5), scales=args.cam_scales)
	train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
								   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)#, worker_init_fn=_init_fn)

	max_step = (len(train_dataset) // args.cam_batch_size) #* args.cam_num_epoches
	model_init_max_step = 5
	model_max_step = max(0, max_step * ((args.cam_num_epoches-5)//2))
	clsbd_max_step = max(0, max_step * ((args.cam_num_epoches-5+1)//2))
	ep_start = 5 if args.cam_preload else 0
	ep_start = max(args.cam_start_epoch, ep_start)
	st_steps = get_st_steps(ep_start,max_step)
	# model init optimizer 
	if not args.cam_preload:
		model_init_param_groups = model.init_trainable_parameters()
		model_init_optimizer = torchutils.PolyOptimizer([
			{'params': model_init_param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
			{'params': model_init_param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
		], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=model_init_max_step, set_step=st_steps[0])
	# model optimizer
	model_param_groups = model.trainable_parameters()
	model_optimizer = torchutils.PolyOptimizer([
		{'params': model_param_groups[0], 'lr': args.seg_learning_rate, 'weight_decay': args.seg_weight_decay},
		{'params': model_param_groups[1], 'lr': 10*args.seg_learning_rate, 'weight_decay': args.seg_weight_decay},
	], lr=args.seg_learning_rate, weight_decay=args.seg_weight_decay, max_step=model_max_step, set_step=st_steps[1])
	# model = torch.nn.DataParallel(model).cuda()

	# clsbd train loader is the same with model loader. 
	# clsbd optimizer 
	clsbd_param_groups = clsbd.trainable_parameters()
	clsbd_optimizer = torchutils.PolyOptimizer([
		{'params': clsbd_param_groups, 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
	], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=clsbd_max_step, set_step=st_steps[2])
	# clsbd = torch.nn.DataParallel(clsbd).cuda()

	avg_meter = pyutils.AverageMeter()
	timer = pyutils.Timer()

	for ep in range(ep_start, args.cam_num_epoches):
		
		# decide what routine to take, model-init, clsbd or model
		print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))
		rt_key = determine_routine(ep,args)
		if rt_key == 'init':
			model_init_train(train_data_loader, model, model_init_optimizer, avg_meter, timer, args)
		elif rt_key == 'model':
			best_miou = 0
			miou = -1
			is_max_step = False
			# model_inf = copy.deepcopy(model)
			while True:
				model_new, is_max_step = model_alternate_train(train_data_loader, model, model_init, clsbd, model_optimizer, avg_meter, timer, args, ep)
				# import pdb;pdb.set_trace()
				miou = model_validate(model_new, args)
				if miou < best_miou or is_max_step:
					break
				best_miou = miou
				model = model_new
		elif rt_key == 'clsbd':
			# clsbd, _ = clsbd_alternate_train(train_data_loader, model, clsbd, clsbd_optimizer, avg_meter, timer, args, ep)
			clsbd_validate(model_init, clsbd, args)

	torch.cuda.empty_cache()
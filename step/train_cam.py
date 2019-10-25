
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

from chainercv.datasets import VOCSemanticSegmentationDataset

from misc.trainutils import determine_routine, clsbd_alternate_train, clsbd_validate, model_alternate_train, model_init_train

def run(args):

	model = getattr(importlib.import_module(args.cam_network), 'Net')()
	if args.cam_preload:
		model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)

	clsbd = getattr(importlib.import_module(args.irn_network), 'Net')()
	if args.clsbd_preload:
		clsbd.load_state_dict(torch.load(args.irn_weights_name + '.pth'), strict=True)

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
	train_dataset = voc12.dataloader.VOC12ClassificationDatasetMSF_Train(args.train_list, voc12_root=args.voc12_root, #resize_long=(320, 640),
																hor_flip=True,
																crop_size=512, crop_method="random",rescale=(1., 1.5), scales=args.cam_scales)
	train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
								   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)#, worker_init_fn=_init_fn)

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
	ep_start = max(args.cam_start_epoch, ep_start)
	for ep in range(ep_start, args.cam_num_epoches):
		
		# decide what routine to take, model-init, clsbd or model
		print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))
		rt_key = determine_routine(ep,args)
		if rt_key == 'init':
			model_init_train(train_data_loader, model, model_optimizer, avg_meter, timer, args)
		elif rt_key == 'model':
			model_alternate_train(train_data_loader, model, clsbd, model_optimizer, avg_meter, timer, args)
		elif rt_key == 'clsbd':
			# clsbd_alternate_train(train_data_loader, model, clsbd, clsbd_optimizer, avg_meter, timer, args)
			clsbd_validate(model.module, clsbd.module, args)

	torch.cuda.empty_cache()
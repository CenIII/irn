
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils
from torch import autograd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np 
from torch.nn.utils import clip_grad_norm_
import os
import random
	
def validate(model, data_loader):
	print('validating ... ', flush=True, end='')

	val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

	model.eval()

	with torch.no_grad():
		for pack in data_loader:
			img = pack['img']
			mask = pack['mask']
			label = pack['label'].cuda(non_blocking=True)

			# x = model(img)
			# loss1 = torchutils.batch_multilabel_loss(x, label)
			preds, pred0, hms, clsbds = model(img, mask, label)
			# loss1 = torchutils.batch_multilabel_loss(preds, label, mean=True)
			loss1 = 0.
			# loss1 = torchutils.multilabel_soft_pull_loss(preds[0], label)#, mean=True)
			# loss1 += F.multilabel_soft_margin_loss(pred0, label)

			val_loss_meter.add({'loss1': loss1}) #.item()

	model.train()

	print('loss: %.4f' % (val_loss_meter.pop('loss1')))

	return

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

def visualize_all_classes(hms, label, iterno, savepath, origin=0, keyword='orig'):
	# plt.figure(2)
	class_name = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train','tvmonitor']

	fig, ax = plt.subplots(nrows=4, ncols=5)
	hms = hms[origin] # if origin else hms[-2]
	N,W,H,C = hms.shape
	for i in range(0, C):
		ax[int(i/5)][int(i%5)].imshow(hms[0][...,i].data.cpu().numpy())
		ax[int(i/5)][int(i%5)].set_title(class_name[i],color='r' if label[0][i]>0 else 'black')
	fig.suptitle('iteration '+str(iterno))
	
	# plt.pause(0.02)
	savename = 'visual_train_'+str(iterno)+'_'+keyword+'_cams.png'
	plt.savefig(os.path.join(savepath, savename))
	plt.close()

def visualize_class_boudaries(clsbds, label, iterno, savepath):
	class_name = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train','tvmonitor']
	# TODO: implement this

	import pdb;pdb.set_trace()
	pass
	return

def run(args):
	model = getattr(importlib.import_module(args.cam_network), 'Net')()
	model.load_state_dict(torch.load('exp/normft_crssent_ftnormap01/sess/res50_cam.pth' + '.pth'), strict=True) #args.cam_weights_name
	
	seed = 42
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
		
	train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
																resize_long=(320, 640), hor_flip=True,
																crop_size=512, crop_method="random")
	train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
								   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
	max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

	val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
															  crop_size=512)
	val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
								 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

	param_groups = model.trainable_parameters()
	optimizer = torchutils.PolyOptimizer([
		{'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
		# {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
		{'params': param_groups[2], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
	], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

	model = torch.nn.DataParallel(model).cuda()
	model.train()
	model.make_bd_weight_dict()

	avg_meter = pyutils.AverageMeter()

	timer = pyutils.Timer()

	cb = [None, None, None, None]
	img_denorm = torchutils.ImageDenorm()
	global_step = 0.
	for ep in range(args.cam_num_epoches):

		print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

		for step, pack in enumerate(train_data_loader):

			img = pack['img'].cuda()
			mask = pack['mask'].cuda()
			label = pack['label'].cuda(non_blocking=True)
			preds, pred0, hms, clsbds = model(img, mask, label)
			if global_step%10 == 0 and args.cam_visualize_train:
				visualize(img, model, hms, label, cb, global_step, img_denorm, args.vis_out_dir)#.module
				visualize_all_classes(hms, label, global_step, args.vis_out_dir)
				visualize_all_classes(hms, label, global_step, args.vis_out_dir, origin=1, keyword='bgap')
				visualize_all_classes(hms, label, global_step, args.vis_out_dir, origin=2, keyword='edge')
				
			loss = 0.
			# loss = torchutils.multilabel_soft_pull_loss(preds[0], label) #, mean=True)
			# loss += F.multilabel_soft_margin_loss(pred0, label)
			# avg_meter.add({'loss1': loss.item()})
			# with autograd.detect_anomaly():
			# 	optimizer.zero_grad()
			# 	loss.backward()
			# 	# import pdb;pdb.set_trace()
			# 	# print(torch.max(model.module.gap.lin.weight.grad))
			# 	clip_grad_norm_(model.parameters(), 1.)
			# 	optimizer.step()

			if global_step%100 == 0:
				timer.update_progress((global_step+1) / max_step)

				print('step:%5d/%5d' % (global_step, max_step),
					  'loss:%.4f' % (avg_meter.pop('loss1')),
					  'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
					  'lr: %.4f' % (optimizer.param_groups[0]['lr']),
					  'etc:%s' % (timer.str_estimated_complete()), flush=True)
			global_step += 1
		else:
			# torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
			validate(model, val_data_loader)
			timer.reset_stage()

	# torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
	torch.cuda.empty_cache()
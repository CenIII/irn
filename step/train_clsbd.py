
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import voc12.dataloader
from misc import pyutils, torchutils, indexing
import importlib
import torch.nn as nn
import numpy as np 
import random
import os 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.utils import clip_grad_norm_
from torch import autograd
from net.resnet50_clsbd import default_conf

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

# def compute_loss(crit,pred,label):
# 	label = label.type(torch.cuda.LongTensor)
# 	N,C,W,H = pred.shape
# 	# pred = torch.log(pred + 1e-5)
# 	pred_flat = pred.permute(0,2,3,1).contiguous().view(-1,C)
# 	label_flat = label.view(-1)
	
# 	bg_inds = torch.nonzero(label_flat==0.).squeeze()
# 	loss = crit(pred_flat[bg_inds],label_flat[bg_inds])
# 	label_flat[label_flat==0.] = 255.
# 	fg_inds = torch.nonzero(label_flat<255.).squeeze()
# 	# import pdb;pdb.set_trace()
	
# 	loss = loss + crit(pred_flat[fg_inds],label_flat[fg_inds])
# 	loss = loss / 2.
# 	return loss

def compute_loss(pred):
	pos, neg, pos_fg_sum, pos_bg_sum, neg_sum = pred
	loss = (pos[:,0]/pos_bg_sum.sum()).sum()/4.+(pos[:,1:]/pos_fg_sum.sum()).sum()/4.+(neg/neg_sum.sum()).sum()/2.
	return loss

def get_grad_norm(parameters, norm_type=2):
	total_norm = 0
	for p in parameters:
		param_norm = p.grad.data.norm(norm_type)
		total_norm += param_norm.item() ** norm_type
	total_norm = total_norm ** (1. / norm_type)
	return total_norm

def run(args):

	path_index = indexing.PathIndex(radius=10, default_size=(args.irn_crop_size // 4, args.irn_crop_size // 4))

	# cam = getattr(importlib.import_module(args.cam_network), 'Net')()
	# cam.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
	# cam.eval()

	model = getattr(importlib.import_module(args.irn_network), 'Net')(default_conf)
	# model = torchutils.reload_model(model, './exp/original_cam/sess/res50_irn.pth')

	irn = getattr(importlib.import_module(args.irn_network), 'Net')(default_conf)
	irn = torchutils.reload_model(irn, './exp/original_cam/sess/res50_irn_orig.pth')

	# model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
	irn.eval()

	seed = 13
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

	train_dataset = voc12.dataloader.VOC12AffinityDataset(args.train_list,
														  label_dir=args.ir_label_out_dir,
														  voc12_root=args.voc12_root,
														  indices_from=path_index.default_src_indices,
														  indices_to=path_index.default_dst_indices,
														  hor_flip=True,
														  crop_size=args.irn_crop_size,
														  crop_method="random",
														  rescale=(0.5, 1.5)
														  )
	train_data_loader = DataLoader(train_dataset, batch_size=args.irn_batch_size,
								   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=_init_fn)

	max_step = (len(train_dataset) // args.irn_batch_size) * args.irn_num_epoches

	param_groups = model.trainable_parameters()
	optimizer = torchutils.PolyOptimizer([
		{'params': param_groups, 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
	], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=max_step)

	model = torch.nn.DataParallel(model).cuda()
	model.train()

	irn = irn.cuda()

	avg_meter = pyutils.AverageMeter()

	timer = pyutils.Timer()
	
	crit = nn.NLLLoss()
	cb = [None, None, None, None]
	img_denorm = torchutils.ImageDenorm()

	for ep in range(args.irn_num_epoches):

		print('Epoch %d/%d' % (ep+1, args.irn_num_epoches))

		for iter, pack in enumerate(train_data_loader):
			img = pack['img'].cuda(non_blocking=True)
			label = pack['reduced_label'].cuda(non_blocking=True)
			cls_label = pack['cls_label'].cuda(non_blocking=True)
			# bg_pos_label = pack['aff_bg_pos_label'].cuda(non_blocking=True)
			# fg_pos_label = pack['aff_fg_pos_label'].cuda(non_blocking=True)
			# neg_label = pack['aff_neg_label'].cuda(non_blocking=True)
			# import pdb;pdb.set_trace()
			with autograd.detect_anomaly():
				pred, hms = model(img, label.clone())
				pred1, hms1 = irn(img[0:1], label.clone()[0:1])
				hms.append(hms1[-1].repeat(img.shape[0],1,1,1))
				# visualization
				if (optimizer.global_step-1)%20 == 0 and args.cam_visualize_train:
					visualize(img, model.module, hms, cls_label, cb, optimizer.global_step-1, img_denorm, args.vis_out_dir)
					visualize_all_classes(hms, cls_label, optimizer.global_step-1, args.vis_out_dir, origin=0, descr='unary')
					visualize_all_classes(hms, cls_label, optimizer.global_step-1, args.vis_out_dir, origin=2, descr='convcrf')
				# TODO: masked pixel cross-entropy loss compute. 
				# import pdb;pdb.set_trace()
				# loss = compute_loss(crit, pred, label)
				loss = compute_loss(pred)
				# loss = - pred.sum()/pred.shape[0]
				
				avg_meter.add({'loss': loss})

				# total_loss = (pos_aff_loss + neg_aff_loss)/2 + (dp_fg_loss + dp_bg_loss)/2
			
				optimizer.zero_grad()
				loss.backward()
				# grad_norm = get_grad_norm(param_groups)
				# clip_grad_norm_(model.parameters(), 2.)
				# import pdb;pdb.set_trace()
				optimizer.step()

			if (optimizer.global_step-1)%100 == 0:
				timer.update_progress(optimizer.global_step / max_step)

				print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
					'loss:%.4f' % (avg_meter.pop('loss')),
					'imps:%.1f' % ((iter+1) * args.irn_batch_size / timer.get_stage_elapsed()),
					# 'grad_norm:%.4f' % grad_norm,
					'lr: %.4f' % (optimizer.param_groups[0]['lr']),
					'etc:%s' % (timer.str_estimated_complete()), flush=True)
		else:
			timer.reset_stage()

	torch.save(model.state_dict(), args.irn_weights_name)
	torch.cuda.empty_cache()

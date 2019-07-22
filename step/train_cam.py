
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
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np 
from torch.nn.utils import clip_grad_norm_
import os
	
def validate(model, data_loader):
	print('validating ... ', flush=True, end='')

	val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

	model.eval()

	with torch.no_grad():
		for pack in data_loader:
			img = pack['img']

			label = pack['label'].cuda(non_blocking=True)

			# x = model(img)
			# loss1 = torchutils.batch_multilabel_loss(x, label)
			preds, pred0, hms = model(img)
			loss1 = torchutils.batch_multilabel_loss(preds, label, mean=True)
			loss1 += F.multilabel_soft_margin_loss(pred0, label)

			val_loss_meter.add({'loss1': loss1.item()})

	model.train()

	print('loss: %.4f' % (val_loss_meter.pop('loss1')))

	return

def visualize(x, net, hms, label, fig, ax, cb, iterno, img_denorm, savepath):
	# import pdb;pdb.set_trace()
	x = img_denorm(x[0].permute(1,2,0).data.cpu().numpy()).astype(np.int32)
	hm = net.getHeatmaps(hms, label.max(dim=1)[1])
	# plot here
	img = ax[0].imshow(x)
	for i in range(len(ax)-1):
		img = ax[i+1].imshow(hm[i][0].data.cpu().numpy())
		if cb[i+1] is not None:
			cb[i+1].remove()
		
		divider = make_axes_locatable(ax[i+1])
		cax = divider.append_axes("right", size="5%", pad=0.05)
		cb[i+1] = plt.colorbar(img, cax=cax)
		# cb[i+1] = plt.colorbar(img, ax=ax[i+1])
	
	fig.suptitle('iteration '+str(iterno))
	
	# plt.pause(0.02)
	plt.savefig(os.path.join(savepath, 'visual_train_'+str(iterno)+'.png'))
	return cb

def run(args):

	model = getattr(importlib.import_module(args.cam_network), 'Net')()


	train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
																resize_long=(320, 640), hor_flip=True,
																crop_size=512, crop_method="random")
	train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
								   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
	max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

	val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
															  crop_size=512)
	val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
								 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

	param_groups = model.trainable_parameters()
	optimizer = torchutils.PolyOptimizer([
		{'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
		{'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
		{'params': param_groups[2], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
	], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

	model = torch.nn.DataParallel(model).cuda()
	model.train()

	avg_meter = pyutils.AverageMeter()

	timer = pyutils.Timer()

	fig, ax = plt.subplots(nrows=1, ncols=3)
	cb = [None, None, None]
	img_denorm = torchutils.ImageDenorm()

	for ep in range(args.cam_num_epoches):

		print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

		for step, pack in enumerate(train_data_loader):

			img = pack['img'].cuda()
			label = pack['label'].cuda(non_blocking=True)

			preds, pred0, hms = model(img)
			if (optimizer.global_step-1)%10 == 0 and args.cam_visualize_train:
				visualize(img, model.module, hms, label, fig, ax, cb, optimizer.global_step-1, img_denorm, args.vis_out_dir)
			loss = torchutils.batch_multilabel_loss(preds, label, mean=True)
			loss += F.multilabel_soft_margin_loss(pred0, label)
			avg_meter.add({'loss1': loss.item()})
			with autograd.detect_anomaly():
				optimizer.zero_grad()
				loss.backward()
				# import pdb;pdb.set_trace()
				# print(torch.max(model.module.gap.lin.weight.grad))
				clip_grad_norm_(model.parameters(), 1.)
				optimizer.step()

			if (optimizer.global_step-1)%100 == 0:
				timer.update_progress(optimizer.global_step / max_step)

				print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
					  'loss:%.4f' % (avg_meter.pop('loss1')),
					  'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
					  'lr: %.4f' % (optimizer.param_groups[0]['lr']),
					  'etc:%s' % (timer.str_estimated_complete()), flush=True)

		else:
			torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
			validate(model, val_data_loader)
			timer.reset_stage()

	torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
	torch.cuda.empty_cache()
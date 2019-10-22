
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils, imutils

from torch import autograd
from torch.nn.utils import clip_grad_norm_
from net.resnet50_clsbd import default_conf

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
	if args.cam_preload:
		ep = ep + 5
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

def make_seg2clsbd_label(seg_output,label,args):
	'''
	Given seg_output of shape ([16, 21, 32, 32]), we make label for clsbd network. 
	return: clsbd_label
	'''
	# 0. detach off
	seg_output = seg_output.detach()
	# 1. upsample
	w,h = seg_output.shape[-2:]
	seg_output = F.interpolate(seg_output, (w*4,h*4), mode='bilinear', align_corners=False) #[16, 21, 128, 128]
	# 2. softmax 
	probs = F.softmax(seg_output,dim=1)
	# 3. obtain & apply argmax mask
	preds = torch.argmax(probs,dim=1).unsqueeze(1)
	mask = probs.data.new(probs.shape).fill_(0.)
	mask = torch.scatter(mask,dim=1,index=preds,value=1.)
	probs_m = probs * mask
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
	return clsbd_label

def compute_clsbd_loss(pred):
	pos, neg, pos_fg_sum, pos_bg_sum, neg_sum = pred
	loss = (pos[:,0]/pos_bg_sum.sum()).sum()/4.+(pos[:,1:]/pos_fg_sum.sum()).sum()/4.+(neg/neg_sum.sum()).sum()/2.
	return loss

def clsbd_alternate_train(train_data_loader, model, clsbd, optimizer, avg_meter, timer, args):
	model.eval()
	clsbd.train()
	for step, pack in enumerate(train_data_loader):
	
		img = pack['img'].cuda()  # ([16, 3, 512, 512])
		label = pack['label'].cuda(non_blocking=True)  # [16, 21]
		seg_output = model(img) # [16, 21, 32, 32]
		seg_label = make_seg2clsbd_label(seg_output, label, args) # format: one channel image, each pixel denoted by class num. 
		import pdb;pdb.set_trace()
		loss_pack, hms = clsbd(img, seg_label)
		loss = compute_clsbd_loss(loss_pack)

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

def run(args):

	model = getattr(importlib.import_module(args.cam_network), 'Net')()
	if args.cam_preload:
		model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
	# TODO: import clsbd network
	clsbd = getattr(importlib.import_module(args.irn_network), 'Net')(default_conf)

	# model train loader
	train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
																resize_long=(320, 640), hor_flip=True,
																crop_size=512, crop_method="random",rescale=(0.5, 1.5))
	train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
								   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
	# TODO: max_step definition?
	max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

	# model optimizer
	model_param_groups = model.trainable_parameters()
	model_optimizer = torchutils.PolyOptimizer([
		{'params': model_param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
		{'params': model_param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
	], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
	model = torch.nn.DataParallel(model).cuda()

	# clsbd train loader is the same with model loader. 
	# clsbd optimizer 
	clsbd_param_groups = clsbd.trainable_parameters()
	clsbd_optimizer = torchutils.PolyOptimizer([
		{'params': clsbd_param_groups, 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
	], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=max_step)
	clsbd = torch.nn.DataParallel(clsbd).cuda()

	avg_meter = pyutils.AverageMeter()
	timer = pyutils.Timer()

	for ep in range(args.cam_num_epoches):
		
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
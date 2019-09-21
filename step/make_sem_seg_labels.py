import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio

import voc12.dataloader
from misc import torchutils, indexing
import tqdm 
from net.resnet50_clsbd import infer_conf

cudnn.enabled = True

def _work(process_id, model, dataset, args, quick=False):

	n_gpus = torch.cuda.device_count()
	if quick:
		databin = dataset#[process_id]
	else:
		databin = dataset[process_id]
	data_loader = DataLoader(databin,
							 shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

	with torch.no_grad(), cuda.device(process_id):

		model.cuda()
		qdar = tqdm.tqdm(enumerate(data_loader),total=len(data_loader),ascii=True,position=process_id)
		for iter, pack in qdar:
			img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
			orig_img_size = np.asarray(pack['size'])

			# edge, dp = model(pack['img'][0].cuda(non_blocking=True))
			# import pdb;pdb.set_trace()
			for k in range(len(pack['img'])):
				pack['img'][k] = pack['img'][k].cuda(non_blocking=True)
			rw = model(pack['img'], pack['unary'].cuda(non_blocking=True), pack['seg_label'].cuda(non_blocking=True))

			cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

			# cams = cam_dict['cam']
			keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

			# cam_downsized_values = cams.cuda()

			# rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
			rw = rw[:,keys]
			rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[0, :, :orig_img_size[0], :orig_img_size[1]]
			rw_up = rw_up / torch.max(rw_up)

			rw_up_bg = F.pad(rw_up[1:], (0, 0, 0, 0, 1, 0), value=0.2)#args.sem_seg_bg_thres
			rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

			rw_pred = keys[rw_pred]
			imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
			imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '_light.png'), (rw_pred*15).astype(np.uint8))
			
			# if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
			# 	print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
	# model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
	cam = getattr(importlib.import_module(args.cam_network), 'Net')()
	cam.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
	cam.eval()

	model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')(cam, infer_conf)
	model = torchutils.reload_model(model, './exp/original_cam/sess/res50_irn_clsbd.pth')

	# model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
	model.eval()

	n_gpus = torch.cuda.device_count()

	if args.quick_make_sem:
		dataset = voc12.dataloader.VOC12ClassificationDatasetMSF_Clsbd(args.quick_list,
															 label_dir=args.ir_label_out_dir,
															 unary_dir=args.unary_out_dir,
															 voc12_root=args.voc12_root,
															 scales=(1.0,0.5))
		_work(0,model,dataset,args,quick=True)
	else:
		dataset = voc12.dataloader.VOC12ClassificationDatasetMSF_Clsbd(args.infer_list,
															 label_dir=args.ir_label_out_dir,
															 unary_dir=args.unary_out_dir,
															 voc12_root=args.voc12_root,
															 scales=(1.0,0.5))
		dataset = torchutils.split_dataset(dataset, n_gpus)

		# print("[", end='')
		multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
		# print("]")
				
	
	
	torch.cuda.empty_cache()

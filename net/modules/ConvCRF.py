"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp
import math

import logging
import warnings

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
					level=logging.INFO,
					stream=sys.stdout)

try:
	import pyinn as P
	has_pyinn = True
except ImportError:
	#  PyInn is required to use our cuda based message-passing implementation
	#  Torch 0.4 provides a im2col operation, which will be used instead.
	#  It is ~15% slower.
	has_pyinn = False
	pass

# from utils import test_utils

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F

import gc


# Default config as proposed by Philipp Kraehenbuehl and Vladlen Koltun,
default_conf = {
	'filter_size': 11,
	'blur': 4,
	'merge': True,
	'norm': 'none',
	'weight': 'vector',
	"unary_weight": 1,
	"weight_init": 0.2,

	'trainable': False,
	'convcomp': True,
	'logsoftmax': True,  # use logsoftmax for numerical stability
	'softmax': True,
	'final_softmax': False,

	'pos_feats': {
		'sdims': 3,
		'compat': 3,
	},
	'col_feats': {
		# 'sdims': 80,
		# 'schan': 13,   # schan depend on the input scale.
		#                # use schan = 13 for images in [0, 255]
		#                # for normalized images in [-0.5, 0.5] try schan = 0.1
		'compat': 10,
		'use_bias': False
	},
	"trainable_bias": False,

	"pyinn": False
}

# Config used for test cases on 10 x 10 pixel greyscale inpu
test_config = {
	'filter_size': 5,
	'blur': 1,
	'merge': False,
	'norm': 'sym',
	'trainable': False,
	'weight': 'scalar',
	"unary_weight": 1,
	"weight_init": 0.5,
	'convcomp': False,

	'trainable': False,
	'convcomp': False,
	"logsoftmax": True,  # use logsoftmax for numerical stability
	"softmax": True,

	'pos_feats': {
		'sdims': 1.5,
		'compat': 3,
	},

	'col_feats': {
		'sdims': 2,
		'schan': 2,
		'compat': 3,
		'use_bias': True
	},
	"trainable_bias": False,
}


class ClsbdCRF(nn.Module):
	""" Implements ConvCRF with hand-crafted features.

		It uses the more generic ConvCRF class as basis and utilizes a config
		dict to easily set hyperparameters and follows the design choices of:
		Philipp Kraehenbuehl and Vladlen Koltun, "Efficient Inference in Fully
		"Connected CRFs with Gaussian Edge Pots" (arxiv.org/abs/1210.5644)
	"""

	def __init__(self, conf, nclasses=None):
		super(ClsbdCRF, self).__init__()

		self.conf = conf
		self.nclasses = nclasses

		self.pos_sdims = 1 / conf['pos_feats']['sdims'] #torch.Tensor([1 / conf['pos_feats']['sdims']])
		# self.col_sdims = None
		# self.col_schan = torch.Tensor([1 / conf['col_feats']['schan']])
		self.col_compat = conf['col_feats']['compat'] #torch.Tensor([conf['col_feats']['compat']])
		self.pos_compat = conf['pos_feats']['compat'] #torch.Tensor([conf['pos_feats']['compat']])

		if conf['weight'] is None:
			weight = None
		elif conf['weight'] == 'scalar':
			val = conf['weight_init']
			weight = torch.Tensor([val]).cuda()
		elif conf['weight'] == 'vector':
			val = conf['weight_init']
			weight = val * torch.ones(1, nclasses, 1, 1).cuda()

		self.CRF = ConvCRF(
			nclasses, mode="col", conf=conf,
			use_gpu=True, filter_size=conf['filter_size'],
			norm=conf['norm'], blur=conf['blur'], trainable=conf['trainable'],
			convcomp=conf['convcomp'], weight=weight,
			final_softmax=conf['final_softmax'],
			unary_weight=conf['unary_weight'],
			pos_weight=conf['pos_weight'],
			neg_weight=conf['neg_weight'],
			pyinn=conf['pyinn'])

		return

	def forward(self, unary, clsbd, label, num_iter=5):
		""" Run a forward pass through ConvCRF.

		Arguments:
			unary: torch.Tensor with shape [bs, num_classes, height, width].
				The unary predictions. Logsoftmax is applied to the unaries
				during inference. When using CNNs don't apply softmax,
				use unnormalized output (logits) instead.

			img: torch.Tensor with shape [bs, 3, height, width]
				The input image. Default config assumes image
				data in [0, 255]. For normalized images adapt
				`schan`. Try schan = 0.1 for images in [-0.5, 0.5]
		"""

		conf = self.conf

		bs, c, x, y = clsbd.shape
		
		pos_feats = self.create_position_feats(clsbd.shape[-2:], sdims=self.pos_sdims, bs=bs)

		compats = [self.col_compat,self.pos_compat]
		is_clsbd_list = [True, False]
		
		self.CRF.add_pairwise_energies([clsbd, pos_feats],
									   compats, is_clsbd_list, conf['merge'])

		prediction = self.CRF.inference(unary, label, num_iter=num_iter)

		self.CRF.clean_filters()
		return prediction

	def _create_mesh(self, shape, requires_grad=False):
		hcord_range = [range(s) for s in shape]
		mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'),
						dtype=np.float32)

		return torch.from_numpy(mesh).cuda()

	# def create_colour_feats(self, img, schan, sdims=0.0, bias=True, bs=1):
	#     norm_img = img * schan

	#     if bias:
	#         norm_mesh = self.create_position_feats(sdims=sdims, bs=bs)
	#         feats = torch.cat([norm_mesh, norm_img], dim=1)
	#     else:
	#         feats = norm_img
	#     return feats

	def create_position_feats(self, shape, sdims, bs=1):
		self.mesh = self._create_mesh(shape)
		return torch.stack(bs * [self.mesh * sdims])


def show_memusage(device=0, name=""):
	import gpustat
	gc.collect()
	gpu_stats = gpustat.GPUStatCollection.new_query()
	item = gpu_stats.jsonify()["gpus"][device]

	logging.info("{:>5}/{:>5} MB Usage at {}".format(
		item["memory.used"], item["memory.total"], name))

def _get_ind(dz):
	if dz == 0:
		return 0, 0
	if dz < 0:
		return 0, -dz
	if dz > 0:
		return dz, 0


def _negative(dz):
	"""
	Computes -dz for numpy indexing. Goal is to use as in array[i:-dz].

	However, if dz=0 this indexing does not work.
	None needs to be used instead.
	"""
	if dz == 0:
		return None
	else:
		return -dz

def polarness(x, label): #[1, 21, 42, 63]
	# import pdb;pdb.set_trace()
	# keys = np.unique(label.cpu())[:-1].astype(np.int32)
	D = x.shape[1] #len(keys) #
	x_t = x#[:,keys]
	# x_t /= x_t.sum(dim=1,keepdim=True)
	entropy = (- x_t * torch.log(x_t+1e-5)).sum(dim=1,keepdim=True)
	if D > 1:
		pl = 1. - entropy / np.log(D)
	else: 
		pl = 1. - entropy
	# pl = 1. - x[:,0:1]
	return pl 

class MessagePassingCol():
	""" Perform the Message passing of ConvCRFs.

	The main magic happens here.
	"""

	def __init__(self, feat_list, compat_list, is_clsbd_list, merge, npixels, nclasses,
				 norm="sym",
				 filter_size=5, clip_edges=0, use_gpu=False,
				 blur=1, matmul=False, verbose=False, pyinn=False):

		assert(use_gpu)

		if not norm == "sym" and not norm == "none":
			raise NotImplementedError

		span = filter_size // 2
		assert(filter_size % 2 == 1)
		self.span = span
		self.filter_size = filter_size
		self.use_gpu = use_gpu
		self.verbose = verbose
		self.blur = blur
		self.pyinn = pyinn

		self.merge = merge

		self.npixels = npixels

		if not self.blur == 1 and self.blur % 2:
			raise NotImplementedError

		self.matmul = matmul

		self._gaus_list = {}
		self._norm_list = []

		def add_gaus(gaus,key):
			tmp = self._gaus_list.setdefault(key,[])
			tmp.append(gaus)
			self._gaus_list[key] = tmp

		for feats, compat, is_clsbd in zip(feat_list, compat_list, is_clsbd_list):
			gaussian = self._create_convolutional_filters(feats, is_clsbd)
			if not norm == "none":
				mynorm = self._get_norm(gaussian)
				self._norm_list.append(mynorm)
			else:
				self._norm_list.append(None)
			
			if is_clsbd:
				add_gaus(compat*(-torch.log(gaussian+1e-5)),'neg')
				add_gaus(-compat*torch.log((1.-gaussian)+1e-5),'pos')
			else:
				add_gaus(compat*gaussian,'neg')
		for k,v in self._gaus_list.items():
			self._gaus_list[k] = sum(v)

	def _get_norm(self, gaus):
		norm_tensor = torch.ones([1, 1, self.npixels[0], self.npixels[1]])
		normalization_feats = torch.autograd.Variable(norm_tensor)
		if self.use_gpu:
			normalization_feats = normalization_feats.cuda()

		norm_out = self._compute_gaussian(normalization_feats, gaussian=gaus)
		return 1 / torch.sqrt(norm_out + 1e-20)
	
	def _get_circle_inds(self,span,i):
		ii = [span-i]*(2*i+1) + list(np.repeat(list(range(span-i+1,span+i)),2)) + [span+i]*(2*i+1)
		jj = list(range(span-i,span+i+1))+[span-i,span+i]*(2*i-1)+list(range(span-i,span+i+1))
		return ii,jj

	def _expand_circle(self,tmp_arr,span,i):
		# if i%2 == 1:
		#     inds = [0]*2+list(range(1,2*i))+[2*i]*2+[0]+[2*i]+list(range(2*i+1,6*i-1))+[6*i-1]+[8*i-1]+[6*i-1]*2 + list(range(6*i,8*i-1))+[8*i-1]*2
		# else:
		#     inds = [0,1] + list(range(1,2*i)) + [2*i-1] + [2*i] + [2*i+1, 2*i+2] + list(range(2*i+1,6*i-1)) + list(range(6*i-3,6*i+1)) + list(range(6*i,8*i-1)) + [8*i-2,8*i-1]
		# tmp_arr = tmp_arr[:,inds]
		inds1 = [0]*2+list(range(1,2*i))+[2*i]*2+[0]+[2*i]+list(range(2*i+1,6*i-1))+[6*i-1]+[8*i-1]+[6*i-1]*2 + list(range(6*i,8*i-1))+[8*i-1]*2
		# inds2 = [0,1] + list(range(1,2*i)) + [2*i-1] + [2*i] + [2*i+1, 2*i+2] + list(range(2*i+1,6*i-1)) + list(range(6*i-3,6*i+1)) + list(range(6*i,8*i-1)) + [8*i-2,8*i-1]
		inds2 = list(range(0,i))+[i]*3+list(range(i+1,2*i+1))+list(range(2*i+1,4*i-1))+[4*i-1,4*i]*3+list(range(4*i+1,6*i-1))+list(range(6*i-1,7*i-1))+[7*i-1]*3+list(range(7*i,8*i))
		tmp_arr1 = tmp_arr[:,inds1]
		tmp_arr2 = tmp_arr[:,inds2]
		tmp_arr = torch.max(tmp_arr1,tmp_arr2)
		return tmp_arr

	def _create_convolutional_filters(self, features, is_clsbd):

		span = self.span

		bs = features.shape[0]

		if self.blur > 1:
			off_0 = (self.blur - self.npixels[0] % self.blur) % self.blur
			off_1 = (self.blur - self.npixels[1] % self.blur) % self.blur
			pad_0 = math.ceil(off_0 / 2)
			pad_1 = math.ceil(off_1 / 2)
			if self.blur == 2:
				assert(pad_0 == self.npixels[0] % 2)
				assert(pad_1 == self.npixels[1] % 2)

			features = torch.nn.functional.avg_pool2d(features,
													  kernel_size=self.blur,
													  padding=(pad_0, pad_1),
													  count_include_pad=False)

			npixels = [math.ceil(self.npixels[0] / self.blur),
					   math.ceil(self.npixels[1] / self.blur)]
			assert(npixels[0] == features.shape[2])
			assert(npixels[1] == features.shape[3])
		else:
			npixels = self.npixels

		gaussian_tensor = features.data.new(
			bs, self.filter_size, self.filter_size,
			npixels[0], npixels[1]).fill_(0)

		gaussian = Variable(gaussian_tensor)

		if is_clsbd:
			# goal: features [1, 1, 86, 125] --> col [1, 7, 7, 86, 125] -tmp_arr-> gaussian [1, 7, 7, 86, 125]
			# 1. make col
			cols = F.unfold(features, self.filter_size, 1, self.span)
			cols = cols.view(bs, self.filter_size, self.filter_size, npixels[0], npixels[1]) #[1, 7, 7, 86, 125]
			# 2. for i in range(span), fill gaussian
			tmp_arr = cols.data.new(bs,8,npixels[0], npixels[1]).fill_(0)

			for i in range(1,span+1):
				# extract ith circle [1,i*8,86,125] from cols
				ii, jj = self._get_circle_inds(span,i)
				src_arr = cols[:,ii,jj]
				# compare with tmp_arr, max to obtain new tmp_arr
				tmp_arr = torch.max(src_arr,tmp_arr)
				# assign tmp_arr to ith circle of gaussian
				update = gaussian.data.new(gaussian.shape).fill_(0.)
				update[:,ii,jj] = tmp_arr
				gaussian = gaussian + update
				# gaussian[:,ii,jj] += tmp_arr
				# expand tmp_arr via index selection
				tmp_arr = self._expand_circle(tmp_arr,span,i)
				# Question: gaussian center element? 
			# gaussian = 1. - gaussian
			gaussian[:,span,span] = 0.
		else:
			for dx in range(-span, span + 1):
				for dy in range(-span, span + 1):

					dx1, dx2 = _get_ind(dx)
					dy1, dy2 = _get_ind(dy)

					feat_t = features[:, :, dx1:_negative(dx2), dy1:_negative(dy2)]
					feat_t2 = features[:, :, dx2:_negative(dx1), dy2:_negative(dy1)] # NOQA

					diff = feat_t - feat_t2
					diff_sq = diff * diff
					exp_diff = torch.exp(torch.sum(-0.5 * diff_sq, dim=1))

					gaussian[:, dx + span, dy + span,
							dx2:_negative(dx1), dy2:_negative(dy1)] = exp_diff

		return gaussian.view(
			bs, 1, self.filter_size, self.filter_size,
			npixels[0], npixels[1])

	def _make_input_col(self,input,label):
		shape = input.shape
		num_channels = shape[1]
		bs = shape[0]
		self.shape = shape

		if self.blur > 1:
			off_0 = (self.blur - self.npixels[0] % self.blur) % self.blur
			off_1 = (self.blur - self.npixels[1] % self.blur) % self.blur
			pad_0 = int(math.ceil(off_0 / 2))
			pad_1 = int(math.ceil(off_1 / 2))
			input = torch.nn.functional.avg_pool2d(input,
												   kernel_size=self.blur,
												   padding=(pad_0, pad_1),
												   count_include_pad=False)
			npixels = [math.ceil(self.npixels[0] / self.blur),
					   math.ceil(self.npixels[1] / self.blur)]
			assert(npixels[0] == input.shape[2])
			assert(npixels[1] == input.shape[3])
		else:
			npixels = self.npixels
		if self.verbose:
			show_memusage(name="Init")

		# if key == 'pos':
		# # polarization as 1. 
		pl = polarness(input, label)  #[1, 1, 42, 63]
		input = input * pl
		if self.pyinn:
			input_col = P.im2col(input, self.filter_size, 1, self.span)
		else:
			# An alternative implementation of num2col.
			#
			# This has implementation uses the torch 0.4 im2col operation.
			# This implementation was not avaible when we did the experiments
			# published in our paper. So less "testing" has been done.
			#
			# It is around ~20% slower then the pyinn implementation but
			# easier to use as it removes a dependency.
			input_unfold = F.unfold(input, self.filter_size, 1, self.span)
			input_unfold = input_unfold.view(
				bs, num_channels, self.filter_size, self.filter_size,
				npixels[0], npixels[1])
			input_col = input_unfold
		return input_col, pl

	def _compute_gaussian(self, input_col, gaussian, norm=None):
		shape = self.shape #input_col.shape
		num_channels = shape[1]
		bs = shape[0]
		# if norm is not None:
		#     input = input * norm

		k_sqr = self.filter_size * self.filter_size

		if self.verbose:
			show_memusage(name="Im2Col")
		
		# if key == 'pos':
		#     product = gaussian * input_col #* pl.unsqueeze(2).unsqueeze(3)
		# else:
		#     product = gaussian * input_col
		# import pdb;pdb.set_trace()
		product = gaussian * input_col

		if self.verbose:
			show_memusage(name="Product")

		product = product.view([bs, num_channels,
								k_sqr, input_col.shape[-2], input_col.shape[-1]])

		message = product.sum(2)

		if self.verbose:
			show_memusage(name="FinalNorm")

		if self.blur > 1:
			in_0 = self.npixels[0]
			in_1 = self.npixels[1]
			off_0 = (self.blur - self.npixels[0] % self.blur) % self.blur
			off_1 = (self.blur - self.npixels[1] % self.blur) % self.blur
			pad_0 = int(math.ceil(off_0 / 2))
			pad_1 = int(math.ceil(off_1 / 2))
			message = message.view(bs, num_channels, input_col.shape[-2], input_col.shape[-1])
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				# Suppress warning regarding corner alignment
				message = torch.nn.functional.upsample(message,
													   scale_factor=self.blur,
													   mode='bilinear')
			message = message[:, :, pad_0:pad_0 + in_0, pad_1:in_1 + pad_1]
			message = message.contiguous()
			message = message.view(shape)
			assert(message.shape == shape)

		# if norm is not None:
		#     message = norm * message

		return message
		
	def compute(self, input, label):
		# if self.merge:
		#     pred = self._compute_gaussian(input, self.gaussian)
		# else:
		#     # assert(len(self._gaus_list) == len(self._norm_list))
		#     pred = 0
		#     for gaus, norm in zip(self._gaus_list, self._norm_list):
		#         pred += self._compute_gaussian(input, gaus, norm)
		input_col, pl = self._make_input_col(input, label)
		preds = {}
		for k,v in self._gaus_list.items():
			preds[k] = self._compute_gaussian(input_col, v)
		return preds, input_col, pl

class ConvCRF(nn.Module):
	"""
		Implements a generic CRF class.

	This class provides tools to build
	your own ConvCRF based model.
	"""

	def __init__(self, nclasses, conf,
				 mode="conv", filter_size=5,
				 clip_edges=0, blur=1, use_gpu=False,
				 norm='sym', merge=False,
				 verbose=False, trainable=False,
				 convcomp=False, weight=None,
				 final_softmax=True, unary_weight=10,
				 pos_weight=1,
				 neg_weight=1,
				 pyinn=False):

		super(ConvCRF, self).__init__()
		self.nclasses = nclasses

		self.filter_size = filter_size
		self.clip_edges = clip_edges
		self.use_gpu = use_gpu
		self.mode = mode
		self.norm = norm
		self.merge = merge
		self.kernel = None
		self.verbose = verbose
		self.blur = blur
		self.final_softmax = final_softmax
		self.pyinn = pyinn

		self.conf = conf

		self.unary_weight = unary_weight

		self.pos_weight = pos_weight
		self.neg_weight = neg_weight

		if self.use_gpu:
			if not torch.cuda.is_available():
				logging.error("GPU mode requested but not avaible.")
				logging.error("Please run using use_gpu=False.")
				raise ValueError

		if trainable:
			def register(name, tensor):
				self.register_parameter(name, Parameter(tensor))
		else:
			def register(name, tensor):
				self.register_buffer(name, Variable(tensor))

		if weight is None:
			self.weight = None
		else:
			register('weight', weight)

		self.neg_comp = nn.Conv2d(nclasses, nclasses,
								kernel_size=1, stride=1, padding=0,
								bias=False)
		# self.comp.weight.data.fill_(0.1 * math.sqrt(2.0 / nclasses))
		self.neg_comp.weight.data = torch.ones_like(self.neg_comp.weight.data)
		for i in range(self.nclasses):
			self.neg_comp.weight.data[i,i] = 0.

	def clean_filters(self):
		self.kernel = None

	def add_pairwise_energies(self, feat_list, compat_list, is_clsbd_list, merge):
		assert(len(feat_list) == len(compat_list))

		assert(self.use_gpu)
		
		npixels = feat_list[0].shape[-2:]
		
		self.kernel = MessagePassingCol(
			feat_list=feat_list,
			compat_list=compat_list,
			is_clsbd_list=is_clsbd_list,
			merge=merge,
			npixels=npixels,
			filter_size=self.filter_size,
			nclasses=self.nclasses,
			use_gpu=True,
			norm=self.norm,
			verbose=self.verbose,
			blur=self.blur,
			pyinn=self.pyinn)

	def inference(self, unary, label, num_iter=3):
		N = unary.shape[0]
		# FIXME: unary must be logits from cam layer. psi_unary = -unary and prediction = softmax(unary)
		# △ 0 Initialize: Q(i.e. prediction) and psi(i.e. psi_unary)
		# import pdb;pdb.set_trace()
		psi_unary = - F.log_softmax(unary, dim=1, _stacklevel=5) #- unary
		# import pdb;pdb.set_trace()
		divs = torch.clamp(unary.view(21,-1).max(dim=1)[0],1.)[None,:,None,None]
		prediction = unary/divs

		# prediction = F.softmax(unary, dim=1)

		norm = False
		for i in range(num_iter):
			# △ 1 Message passing
			messages, input_col, pl = self.kernel.compute(prediction, label)
			_,C,K,_,W,H = input_col.shape
			# △ 2 Compatibility transform
			# mle setting
			# message normalize over polarized points, kernel wise.
			if norm: 
				# import pdb;pdb.set_trace()
				kernel_norm = torch.clamp(input_col.sum(dim=2).sum(dim=2),1.).detach()
				pos_message = messages['pos']/kernel_norm
				kernel_norm_neg = torch.clamp(self.neg_comp(input_col.view(N,C,-1,1)).view(input_col.shape).sum(dim=2).sum(dim=2),1.).detach()
				# messages['neg'] = messages['neg']/kernel_norm
				neg_message = self.neg_comp(messages['neg'])/kernel_norm_neg
			else:
				pos_message = messages['pos']
				neg_message = self.neg_comp(messages['neg'])
				
			# △ 3 Local Update (and normalize)
			# import pdb;pdb.set_trace()
			if self.training:
				pl_pred = (prediction*pl)[:,:,None,None].detach()

				pos_norm = (pl_pred*input_col).view(N,C,-1).sum(dim=2)[:,:,None,None]
				pos_norm[:,1:] = pos_norm[:,1:].sum(dim=1,keepdim=True)
				# pos_norm *= 2.
				pos_norm = torch.clamp(pos_norm, 1.).detach()
				neg_input_col = self.neg_comp(input_col.view(N,C,-1,1)).view(input_col.shape)
				neg_norm = torch.clamp((pl_pred*neg_input_col).view(N,-1).sum(dim=1)[:,None,None,None],1.).detach()

				

				# if self.weight is None:
				#     prediction = - psi_unary - pos_message - neg_message
				# else:
				# import pdb;pdb.set_trace()
				prediction = - (self.unary_weight - self.weight) * psi_unary - self.weight * (self.pos_weight*pos_message/pos_norm + self.neg_weight*neg_message/neg_norm)
				prediction = (prediction*pl_pred.squeeze())#.view(N,-1).sum(dim=1)
				return prediction
			prediction = - (self.unary_weight - self.weight) * psi_unary - self.weight * (self.pos_weight*pos_message + self.neg_weight*neg_message)
			prediction = F.softmax(prediction, dim=1)
			# if not i == num_iter - 1 or self.final_softmax:
			#     if self.conf['softmax']:
			# prediction = prediction*pl_pred#F.softmax(prediction*pl_pred, dim=1)
		# prediction = (prediction*pl_pred.squeeze())#.view(N,-1).sum(dim=1)
		# prediction = - (self.unary_weight - self.weight) * psi_unary - self.weight * (self.pos_weight*pos_message/pos_norm + self.neg_weight*neg_message/neg_norm2)
		return prediction#, loss


def get_test_conf():
	return test_config.copy()


def get_default_conf():
	return default_conf.copy()

# if __name__ == "__main__":
#     conf = get_test_conf()
#     tcrf = GaussCRF(conf, [10, 10], None).cuda()

#     unary = test_utils._get_simple_unary()
#     img = test_utils._get_simple_img()

#     img = np.transpose(img, [2, 0, 1])
#     img_torch = Variable(torch.Tensor(img), requires_grad=False).cuda()

#     unary_var = Variable(torch.Tensor(unary)).cuda()
#     unary_var = unary_var.view(2, 10, 10)
#     img_var = Variable(torch.Tensor(img)).cuda()

#     prediction = tcrf.forward(unary_var, img_var).cpu().data.numpy()
#     res = np.argmax(prediction, axis=0)
#     import scipy.misc
#     scp.misc.imsave("out.png", res)
#     # d.addPairwiseBilateral(2, 2, img, 3)

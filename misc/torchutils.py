import torch
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import math


from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
	def __init__(self, optimizer, step_size, iter_max, power, last_epoch=-1):
		self.step_size = step_size
		self.iter_max = iter_max
		self.power = power
		super(PolynomialLR, self).__init__(optimizer, last_epoch)
		self.start_step = self.last_epoch

	def polynomial_decay(self, lr):
		return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power
	
	def is_max_step(self):
		return (self.iter_max - self.last_epoch) < 5
	
	def set_start_step(self,step):
		self.start_step = step
	def get_lr(self):
		if (
			(self.last_epoch == 0)
			or (self.last_epoch % self.step_size != 0)
			or (self.last_epoch > self.iter_max)
		):
			return [group["lr"] for group in self.optimizer.param_groups]
		return [self.polynomial_decay(lr) for lr in self.base_lrs]

class PolyOptimizer(torch.optim.SGD):

	def __init__(self, params, lr, weight_decay, max_step, momentum=0.9, set_step=0):
		super().__init__(params, lr, weight_decay)

		self.global_step = set_step
		self.max_step = max_step
		self.momentum = momentum

		self.__initial_lr = [group['lr'] for group in self.param_groups]

	def is_max_step(self):
		return (self.max_step - self.global_step) < 5

	def step(self, closure=None):

		if self.global_step < self.max_step:
			lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

			for i in range(len(self.param_groups)):
				self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

		super().step(closure)

		self.global_step += 1

class SGDROptimizer(torch.optim.SGD):

	def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
		super().__init__(params, lr, weight_decay)

		self.global_step = 0
		self.local_step = 0
		self.total_restart = 0

		self.max_step = steps_per_epoch * epoch_start
		self.restart_mult = restart_mult

		self.__initial_lr = [group['lr'] for group in self.param_groups]


	def step(self, closure=None):

		if self.local_step >= self.max_step:
			self.local_step = 0
			self.max_step *= self.restart_mult
			self.total_restart += 1

		lr_mult = (1 + math.cos(math.pi * self.local_step / self.max_step))/2 / (self.total_restart + 1)

		for i in range(len(self.param_groups)):
			self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

		super().step(closure)

		self.local_step += 1
		self.global_step += 1


def split_dataset(dataset, n_splits):

	return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
	out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
	if keepdims:
		out = out.view(out.size(0), out.size(1), 1, 1)

	return out

def leaky_log(x, leaky_rate=0.2):
	hm = torch.sqrt(F.relu(x)) #torch.log(1 + F.relu(x))
	hm_lky = hm - leaky_rate * F.relu(-x)
	return hm_lky

def reload_model(model, path):
	checkpoint = torch.load(path)
	# model.load_state_dict(self.checkpoint['state_dict'])
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {}
	for k, v in checkpoint.items():
		if 'module' in k:
			k = k[7:]
		if(k in model_dict):
			pretrained_dict[k] = v
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	model.load_state_dict(model_dict)
	checkpoint = None
	return model 
	
class ImageDenorm():
	def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
		self.mean = mean
		self.std = std

	def __call__(self, img):
		imgarr = np.asarray(img)
		proc_img = np.empty_like(imgarr, np.float32)

		proc_img[..., 0] = (imgarr[..., 0] * self.std[0] + self.mean[0]) * 255.
		proc_img[..., 1] = (imgarr[..., 1] * self.std[1] + self.mean[1]) * 255.
		proc_img[..., 2] = (imgarr[..., 2] * self.std[2] + self.mean[2]) * 255.

		return proc_img
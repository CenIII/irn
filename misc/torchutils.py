import torch
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import math

if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


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

def batch_multilabel_loss(preds, label, mean=False):
    loss = []
    for pred in preds:
        loss.append(F.multilabel_soft_margin_loss(pred, label))
    ret = sum(loss)
    if mean:
        ret = ret/len(loss)
    return ret 

def multilabel_soft_pull_loss(input, target, weight=None,reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
    r"""multilabel_soft_margin_loss(input, target, weight=None, size_average=None) -> Tensor
    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    # if size_average is not None or reduce is not None:
    #     reduction = _Reduction.legacy_get_string(size_average, reduce)

    loss = -(target * torch.log(torch.sigmoid((input)))) #+ (1 - target) * logsigmoid(-input))

    if weight is not None:
        loss = loss * weight

    loss = loss.sum(dim=1) / input.size(1)  # only return N loss values

    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        ret = loss.mean()
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret
    
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1, dilate=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding / dilate - field_height) % stride == 0
    assert (W + 2 * padding / dilate - field_height) % stride == 0
    out_height = int((H + 2 * padding / dilate - field_height) / stride + 1)
    out_width = int((W + 2 * padding / dilate - field_width) / stride + 1)

    i0 = np.repeat(np.arange(0,field_height*dilate,dilate), field_width) #(9,)
    i0 = np.tile(i0, C) #(1152,)
    i1 = stride * np.repeat(np.arange(out_height), out_width) #(3844,)
    j0 = np.tile(np.arange(0,field_width*dilate,dilate), field_height * C)  #(1152,)
    j1 = stride * np.tile(np.arange(out_width), out_height) #(3844,)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1) #(1152, 3844)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1) #(1152, 3844)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1) #(1152, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1, dilate=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = F.pad(x, (p, p, p, p), mode='constant', value=0)

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride, dilate)

    cols = x_padded[:, k, i, j] #torch.Size([8, 800, 15625])
    C = x.shape[1]
    cols = cols.permute(1, 2, 0).contiguous().view(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1, dilate=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = torch.zeros((N, C, H_padded, W_padded)).type(device.FloatTensor)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                stride,dilate)
    cols_reshaped = cols.view(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.permute(2, 0, 1) #[4, 576, 15376]
    # np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    ll = torch.from_numpy(np.arange(N))[:,None,None].repeat(1,C* field_height * field_width,H*W).view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]
    kk = torch.from_numpy(k)[None,:,:].repeat(N,1,H*W).view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]
    ii = torch.from_numpy(i)[None,:,:].repeat(N,1,1).view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]
    jj = torch.from_numpy(j)[None,:,:].repeat(N,1,1).view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]
    cols_sliced = cols_reshaped.view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]

    for pt in range(field_height * field_width):
        x_padded[ll[pt],kk[pt],ii[pt],jj[pt]] += cols_sliced[pt]
    
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

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
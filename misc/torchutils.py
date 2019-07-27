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
    x_padded = F.pad(x, (p, p, p, p), mode='constant',value=0)

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

def im2col_boundary(x, field_height, field_width, padding=1, stride=1, dilate=1,dist_to_center=None):
    # TODO: 1. check dist_to_center.
    # 2. Modify the cols_list generation
    # 3. Modify search range

    # For boundary map, D=1.
    # dist_to_center [ksize,ksize]
    cols_list = [] 
    cols_max_list = []
    
    # ============= collect the path from center to dilate*int(ksize/2)
    # (e.g: ksize=5,dilate=2. Then collect point with distance [0,4])

    # import pdb;pdb.set_trace()
    # deal with center, reuse code for dilate=1
    padding = int(field_height/2)*1
    p = padding
    x_padded = F.pad(x, (p, p, p, p), mode='constant',value=0)
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                stride, 1)
    cols_i = x_padded[:, k, i, j] #torch.Size([B, fW*fH*D, W*H])
    # cols_i = cols_i[:,int(field_height*field_width/2),:].unsqueeze(1) # copy the value of center point
    cols_i = cols_i.expand(-1,field_height*field_width,-1)
    cols_list.append(cols_i)

    cols_i = cols_i.expand(-1,field_height*field_width,-1)
    cols_list.append(cols_i)

    for dilate_i in range(1,int(field_height/2)*dilate+1): # 1~int(field_height/2)*dilate
        padding = int(field_height/2)*dilate_i # assume the height& width is always the same
        p = padding
        x_padded = F.pad(x, (p, p, p, p), mode='constant',value=0)
        # import pdb;pdb.set_trace()
        k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                    stride, dilate_i)
        cols_i = x_padded[:, k, i, j] #torch.Size([B, fW*fH*D, W*H])
        cols_list.append(cols_i)
    cols_all = torch.stack(cols_list) # torch.Size([Dilate, B, fW*fH*D, W*H])

    # For each distance, find the max value on path with dilationd distance [0,dist_i*dilate]
    for dist_i in range(0,int(field_height/2)+1): # calculate max on path, dilate_i=k choose from k+1 values
        cols,_ = torch.max(cols_all[:dist_i*dilate+1,:,:,:],dim=0) # torch.Size([B, fW*fH*D, W*H])
        cols_max_list.append(cols)
    cols_max_all = torch.stack(cols_max_list) # torch.Size([Dilate, B, fW*fH*D, W*H])

    # todo: 
    ksize = field_height
    dist_to_center = torch.zeros([ksize,ksize])
    mid = int(ksize/2)
    for i in range(ksize):
        for j in range(ksize):
            dist_to_center[i][j]= max(abs(i-mid),abs(j-mid))
    dist_to_center = dist_to_center.type(device.LongTensor)
    
    dist_to_center = dist_to_center.reshape(1,1,-1,1).expand(1,cols_max_all.shape[1],-1,cols_max_all.shape[3]).contiguous() # torch.Size([1, B, fW*fH*D, W*H])
    cols = torch.gather(cols_max_all,dim=0,index=dist_to_center).squeeze(0) # !!the problem occurs here
    C = x.shape[1] #D
    # [5,mid] is the original pixel for (2,2)
    
    cols = cols.permute(1, 2, 0).contiguous().view(field_height * field_width * C, -1) # [fW*fH*D,W*H*B] (feature for each pixel ,num of pixels)
    return cols

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
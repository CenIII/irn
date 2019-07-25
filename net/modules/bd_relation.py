import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from misc.torchutils import im2col_indices, col2im_indices, im2col_boundary
if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device
import torch.nn.functional as F

class Infuse(nn.Module):
    def __init__(self, v_dim, kernel_size=5, dilation=3):
        super(Infuse, self).__init__()  
        self.v_dim = v_dim

    def forward(self,V,boundary,ksize,dilation,dist_to_center):
        # get a [1,1,fW*fH,W*H*B] attention 
        
        N, D, H, W = V.shape
        padding = int(ksize/2)*dilation
        Hf = ksize
        Wf = ksize
        # import pdb;pdb.set_trace()
        boundary_max = im2col_boundary(boundary, Hf, Wf, padding, 1, dilation,dist_to_center)
        boundary_sim = torch.ones_like(boundary_max)-boundary_max+1e-5
        boundary_sim = boundary_sim.unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 25, 52488])
        
        att =  boundary_sim/torch.sum(boundary_sim,dim=2,keepdim=True) #torch.softmax(boundary_sim, 2) # [1,1,fW*fH,W*H*B]
        att = att*((1-boundary).permute(1,2,3,0).contiguous().view(-1)[None,None,None,:])
        V_trans = im2col_indices(V, Hf, Wf, padding, 1, dilation).view(1, self.v_dim, Hf*Wf, -1) # torch.Size([1, 2, 9, 52488])
        out = (V_trans * att).sum(2).sum(0).view(D, H, W, N).permute(3, 0, 1, 2) # torch.Size([8, 2, 81, 81])
        return out


class BDInfusion(nn.Module):
    def calculate_dist_pattern(self,rel_pattern):
        # use_cuda = torch.cuda.is_available()
        pattern_dict = {}
        for ksize, _ in rel_pattern:
            pattern = torch.zeros([ksize,ksize])
            mid = int(ksize/2)
            for i in range(ksize):
                for j in range(ksize):
                    pattern[i][j]= max(abs(i-mid),abs(j-mid))
            pattern = pattern.type(device.LongTensor)
            # if use_cuda:
                # pattern=pattern.cuda()
            pattern_dict[ksize]=pattern
        return pattern_dict
        
    def __init__(self, in_channels, n_class, rel_pattern=[(5, 3)]):
        super(BDInfusion, self).__init__()
        self.rel_pattern = rel_pattern
        self.infuse = Infuse(in_channels)
        self.n_class = n_class
        self.dist_pattern_dict = self.calculate_dist_pattern(rel_pattern)

    def forward(self, feats, boundary):
        N = feats.shape[0]
        feats_r = []
        # import pdb;pdb.set_trace()
        for ksize, dilation in self.rel_pattern:
            # dist_to_center = self.dist_pattern_dict[ksize]
            feats_r.append(self.infuse(feats,boundary, ksize, dilation,None))#self.dist_pattern_dict[ksize]))
            
        feats_r = torch.stack(feats_r, dim=0).sum(0)
        pred_r = torch.mean(feats_r.view(N, self.n_class, -1), dim=2)
        return pred_r, feats_r
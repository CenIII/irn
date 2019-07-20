import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tools import im2col_indices, col2im_indices
if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device
import torch.nn.functional as F

class Gap(nn.Module):
    def __init__(self, in_channels, n_class):
        super(Gap, self).__init__()
        self.lin = nn.Conv2d(in_channels,n_class,1,bias=False) #nn.Linear(in_channels, n_class, bias=False)
        self.n_class = n_class

    def forward(self, x):
        N = x.shape[0]
        cam = self.lin(x) #.permute(0, 2, 3, 1)
        pred = torch.mean(cam.view(N, self.n_class, -1), dim=2)
        return pred, cam

    def infer_class_maps(self, x):
        x = self.lin(x.permute(0, 2, 3, 1))
        return x

    def make_mask(self, label):
        N, W, H, _ = self.heatmaps.shape
        soft_mask = F.relu(self.heatmaps)
        mask = torch.gather(soft_mask, 3, label[:, None, None, None].repeat(1, W, H, 1)).squeeze().detach() # (8, 81, 81)
        mask_norm = (mask/torch.clamp(torch.max(mask.view(N, -1), dim=1)[0], 1.)[:, None, None]).unsqueeze(1)
        mask_norm[mask_norm>0.15]=1
        mask_norm[mask_norm<=0.15]=0
        return mask_norm

class LeakyLogGap(Gap):
    def __init__(self, in_channels, n_class):
        super(LeakyLogGap, self).__init__(in_channels, n_class)
    def forward(self, x, save_hm=False):
        N = x.shape[0]
        pre_hm = self.lin(x.permute(0, 2, 3, 1))
        x = torch.log(1 + F.relu(pre_hm))
        if save_hm:
            self.heatmaps = x
        x = torch.mean((x - 0.12 * F.relu(-pre_hm)).view(N, -1, self.n_class), dim=1)
        return x

class KQ(nn.Module):
    def __init__(self, in_channels, kq_dim):
        super(KQ, self).__init__()
        self.k = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(in_channels, kq_dim, (1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(kq_dim, kq_dim, (1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(kq_dim, kq_dim, (1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(kq_dim, kq_dim, (1, 1)), 
            )
        self.q = nn.Sequential(
            nn.Conv2d(in_channels, kq_dim, (1, 1)), 
            )
        
    def forward(self, x):
        K = self.k(x)
        Q = self.q(x)
        return K, Q

    
class Infusion(nn.Module):
    def __init__(self, v_dim, kq_dim, n_heads=1, kernel_size=5, dilation=3):
        super(Infusion, self).__init__()  
        assert(kq_dim % n_heads == 0)
        self.v_dim = v_dim
        self.kq_dim = kq_dim
        self.n_heads = n_heads
        self.h_dim = int(kq_dim / n_heads)

    def forward(self, V, K, Q, ksize, dilation): # NxDxHxW
        N, D, H, W = V.shape # 8, 32, 124, 124
        padding = int(ksize/2)*dilation
        Hf = ksize
        Wf = ksize
        K_trans = im2col_indices(K, Hf, Wf, padding, 1, dilation) # (3200, 38440)
        Q_trans = Q.permute(1, 2, 3, 0).contiguous().view(self.n_heads, self.h_dim, -1) # (128, 38440)
        tmp = (K_trans.view(self.n_heads, self.h_dim, -1, K_trans.shape[-1]) - Q_trans.unsqueeze(2)).view(self.n_heads, self.h_dim, Hf*Wf, -1)
        tmp = torch.pow(tmp,2).sum(1, True)/np.sqrt(self.h_dim) # (4, 1, 5*5, 38440) 
        att = torch.softmax(-tmp, 2) # (4, 1, 25, 38440)
        V_trans = im2col_indices(V, Hf, Wf, padding, 1, dilation).view(1, self.v_dim, Hf*Wf, -1)
        out = (V_trans * att).sum(2).sum(0).view(D, H, W, N).permute(3, 0, 1, 2)/self.n_heads
        return out

class Diffusion(nn.Module):
    def __init__(self, v_dim, kq_dim, n_heads=1):
        super(Diffusion, self).__init__()  
        assert(kq_dim % n_heads == 0)
        self.v_dim = v_dim
        self.kq_dim = kq_dim
        self.n_heads = n_heads
        self.h_dim = int(kq_dim/n_heads)

    def get_att_map(self):
        att_cols=self.att.view(Hf*Wf, -1).detach()
        att_new=col2im_indices(att_cols, [N, self.n_heads, H, W], Hf, Wf, padding, 1, dilation)
        att_new = torch.clamp(att_new, 2.)
        return att_new

    def forward(self, V, K, Q, ksize, dilation, mask):
        N, D, H, W = Q.shape # 8, 32, 124, 124
        Dv = V.shape[1]
        Hf = ksize
        Wf = ksize
        padding = int(ksize/2)*dilation
        
        K_trans = im2col_indices(K, Hf, Wf, padding, 1, dilation) # (800, 123008)
        Q_trans = Q.permute(1, 2, 3, 0).contiguous().view(D, -1) # (32, 123008)
        tmp = (K_trans.view(D, -1, K_trans.shape[-1]) * Q_trans.unsqueeze(1)).view(self.n_heads, self.h_dim, Hf*Wf, -1)
        tmp = tmp.sum(1, True)/np.sqrt(self.h_dim) # (4, 1, 5*5, 123008)
        self.att = torch.softmax(tmp, 2) # (4, 1, 25, 123008)
        mask = mask.squeeze().permute(1, 2, 0).contiguous().view(-1)
        self.att = self.att * mask[None, None, None, :] 

        V_trans = V.permute(1, 2, 3, 0).contiguous().view(self.n_heads, int(Dv/self.n_heads), 1, H*W*N) # (4, h_dim, 1, 123008)
        V_cols = (V_trans * self.att).view(Dv*Hf*Wf, -1) # (n_heads*h_dim*25, 123008)
        V_new = col2im_indices(V_cols, V.shape, Hf, Wf, padding, 1, dilation)
        return V_new

class Relation(nn.Module):
    def __init__(self, in_channels, kq_dim, n_class, n_heads=1, rel_pattern=[(5, 3)]):
        super(Relation, self).__init__()
        self.rel_pattern = rel_pattern
        self.infuse = Infusion(in_channels, kq_dim, n_heads=n_heads)
        self.n_class = n_class
        
    def forward(self, feats, K, Q):
        N = feats.shape[0]
        feats_r = []
        for ksize, dilation in self.rel_pattern:
            feats_r.append(self.infuse(feats, K, Q, ksize, dilation))
        feats_r = torch.stack(feats_r, dim=0).sum(0)
        pred_r = torch.mean(feats_r.view(N, self.n_class, -1), dim=2)
        return pred_r, feats_r

        

        
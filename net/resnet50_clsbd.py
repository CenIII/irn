import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50
from net.modules import ClsbdCRF
import numpy as np

# Default config as proposed by Philipp Kraehenbuehl and Vladlen Koltun,
default_conf = {
    'filter_size': 19,
    'blur': 1,
    'merge': False,
    'norm': 'none',
    'weight': 'vector',
    "unary_weight": 1.,
    "weight_init": 1.,
    "pos_weight":1.,
    "neg_weight":1.,

    'trainable': False,
    'convcomp': True,
    'logsoftmax': True,  # use logsoftmax for numerical stability
    'softmax': True,
    'final_softmax': False,

    'pos_feats': {
        'sdims': 50,
        'compat': 0.,
    },
    'col_feats': {
        # 'sdims': 80,
        # 'schan': 13,   # schan depend on the input scale.
        #                # use schan = 13 for images in [0, 255]
        #                # for normalized images in [-0.5, 0.5] try schan = 0.1
        'compat': 1.,
        'use_bias': False
    },
    "trainable_bias": False,

    "pyinn": False
}

infer_conf = {
    'filter_size': 11,
    'blur': 2,
    'merge': False,
    'norm': 'none',
    'weight': 'vector',
    "unary_weight": 1.,
    "weight_init": 0.9,
    "pos_weight":20.,
    "neg_weight":1.,

    'trainable': False,
    'convcomp': True,
    'logsoftmax': True,  # use logsoftmax for numerical stability
    'softmax': True,
    'final_softmax': False,

    'pos_feats': {
        'sdims': 100,
        'compat': 1.,
    },
    'col_feats': {
        # 'sdims': 80,
        # 'schan': 13,   # schan depend on the input scale.
        #                # use schan = 13 for images in [0, 255]
        #                # for normalized images in [-0.5, 0.5] try schan = 0.1
        'compat': 1.,
        'use_bias': False
    },
    "trainable_bias": False,

    "pyinn": False
}


class Net(nn.Module):

    def __init__(self, crf_conf):
        super(Net, self).__init__()
        # self.cam_net = cam_net
        # backbone
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=[2, 2, 2, 1])

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)

        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        self.convcrf = ClsbdCRF(crf_conf, nclasses=21)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])

    def infer_clsbd(self, x): # no sigmoid
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()

        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        edge_up = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        edge_out = edge_up
        return edge_out

    def increase_contrast(self,unary_norm,factor=2.):
        return torch.clamp(factor*(unary_norm-0.5)+0.5,0.,1.)

    def make_unary_for_infer(self, unary_raw, label):
        # unary_raw [N, 21, W, H]
        # 1. rescale
        # unary_raw = F.interpolate(unary_raw, label.shape[-2:], mode='bilinear', align_corners=False)#[0] #torch.unsqueeze(unary_raw, 0)
        # 2. add background
        # unary_raw /= 5.
        W, H = unary_raw.shape[-2:]
        # import pdb;pdb.set_trace()
        keys = label.squeeze().nonzero()[:,0] #torch.unique(label)[1:-1]
        mask = torch.zeros_like(unary_raw).cuda()
        for k in keys:
            mask[:,int(k)] = 1.
        unary = (unary_raw * mask)
        unary_norm = unary / torch.clamp(F.adaptive_max_pool2d(unary, (1, 1)),1)
        # unary_norm = self.increase_contrast(unary_norm,factor=1.5)
        # unary_norm[:,0] = self.increase_contrast(unary_norm[:,0],factor=1.5)
        # unary_norm[:,3] = self.increase_contrast(unary_norm[:,3],factor=1.5)
        # unary_norm[:,18] = self.increase_contrast(unary_norm[:,18],factor=1.5)
        unary = F.pad(unary, (0, 0, 0, 0, 1, 0, 0, 0), mode='constant',value=1.)
        unary_norm = F.pad(unary_norm, (0, 0, 0, 0, 1, 0, 0, 0), mode='constant',value=0.08)
        pred = torch.argmax(unary_norm, dim=1)
        pred = pred.unsqueeze(1)
        mask = torch.zeros_like(unary_norm).cuda()
        mask = mask.scatter_(1,pred.type(torch.cuda.LongTensor),1.)
        unary = (unary * mask)
        unary = unary / torch.clamp(F.adaptive_max_pool2d(unary, (1, 1)),1)
        # unary[:,1:] = unary[:,1:]*2.
        # unary[:,0] = unary[:,0]*1

        # 3. create and apply mask
        # label[label==255.] = 21
        # label = label.unsqueeze(1)
        # mask = torch.zeros_like(unary_raw).cuda()
        # mask = mask.scatter_(1,label.type(torch.cuda.LongTensor),1.)
        # unary = (unary_raw * mask)[:,:-1]
        # unary[unary>0.] = 150.
        # tmp = mask[:,:-1].sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) # [N,21,1,1]
        # tmp[tmp>0.] = 1.
        # unary += tmp
        # unary[:,0] = 1.
        return unary 

    def make_unary_for_train(self, label):
        # unary_raw [N, 21, W, H]
        # 1. rescale
        # unary_raw = F.interpolate(unary_raw, label.shape[-2:], mode='bilinear', align_corners=False)#[0] #torch.unsqueeze(unary_raw, 0)
        # # 2. add background
        # unary_raw = F.pad(unary_raw, (0, 0, 0, 0, 1, 1, 0, 0), mode='constant',value=1.)
        # 3. create and apply mask
        label[label==255.] = 21
        label = label.unsqueeze(1)
        N,_,W,H = label.shape
        mask = torch.zeros(N,22,W,H).cuda()
        mask = mask.scatter_(1,label.type(torch.cuda.LongTensor),1.)
        # unary = (unary_raw * mask)[:,:-1]
        unary = mask[:,:-1]
        unary[unary>0.] = 100.
        return unary 

    def forward(self, x, label):
        # unary_raw = self.cam_net(x)
        # unary_raw = F.relu(unary_raw).detach()
        unary = self.make_unary_for_train(label)
        clsbd = self.infer_clsbd(x)[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd = torch.sigmoid(clsbd)
        pred = self.convcrf(unary, clsbd, label, num_iter=1)
        hms = self.save_hm(unary,clsbd.repeat(1,21,1,1))
        return pred, hms

    def getHeatmaps(self, hms, classid):
        hm = []
        for heatmap in hms:
            zzz = classid[:, None, None, None].repeat(1, heatmap.shape[1], heatmap.shape[2], 1)
            hm.append(torch.gather(heatmap, 3, zzz).squeeze())
        return hm

    def save_hm(self, *cams):
        hm = []
        for cam in cams:
            hm.append(cam.permute(0,2,3,1))
        return hm

    def trainable_parameters(self):

        return (tuple(self.edge_layers.parameters()))#, tuple(self.dp_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


class Sobel:
    def __init__(self):

        self.a = torch.Tensor( [[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]]).view((1,1,3,3)).cuda()

        self.b = torch.Tensor( [[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]]).view((1,1,3,3)).cuda()
        self.blob = torch.Tensor(  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]).view((1,1,3,3)).cuda()*20

        self.dir_filt = torch.Tensor([[ [1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]],
                                        [[0, 1, 0],
                                        [0, 1, 0],
                                        [0, 1, 0]],
                                        [[0, 0, 1],
                                        [0, 1, 0],
                                        [1, 0, 0]],
                                        [[0, 0, 0],
                                        [1, 1, 1],
                                        [0, 0, 0]]]).view((4,1,3,3)).cuda()
    # def filt(self,x):
    #     G_x = F.conv2d(x, self.a, padding=1)
    #     G_y = F.conv2d(x, self.b, padding=1)
    #     G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    #     G = G / G.max() * 255
    #     theta = torch.atan2(G_y, G_x)
    #     return (G, theta)
    
    # def nms(self, img, D):
    #     img = img.squeeze()
    #     M, N = img.shape[-2:]
    #     Z = torch.zeros((M,N)).cuda()
    #     angle = D * 180. / np.pi
    #     angle[angle < 0] += 180
    #     angle = angle.squeeze()
        
    #     import pdb;pdb.set_trace()

    #     for i in range(1,M-1):
    #         for j in range(1,N-1):
    #             q = 255
    #             r = 255
                
    #             #angle 0
    #             if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
    #                 q = img[i, j+1]
    #                 r = img[i, j-1]
    #             #angle 45
    #             elif (22.5 <= angle[i,j] < 67.5):
    #                 q = img[i+1, j-1]
    #                 r = img[i-1, j+1]
    #             #angle 90
    #             elif (67.5 <= angle[i,j] < 112.5):
    #                 q = img[i+1, j]
    #                 r = img[i-1, j]
    #             #angle 135
    #             elif (112.5 <= angle[i,j] < 157.5):
    #                 q = img[i-1, j-1]
    #                 r = img[i+1, j+1]

    #             if (img[i,j] >= q) and (img[i,j] >= r):
    #                 Z[i,j] = img[i,j]
    #             else:
    #                 Z[i,j] = img[i,j]/2.
    #     return Z
    def nms(self, x):
        N,C,W,H = x.shape
        x_unfold = F.unfold(x, 3, 1, 1).view(N,C,3,3,W,H)
        # 2. check every point whether it's a peak on any direction? gen a mask, 1 if satisfy.
        x_diff = x_unfold - x[:,:,None,None]
        mask1 = x_diff.data.new(x_diff.shape).fill_(0)
        mask1[x_diff<0.] = 1.
        mask2 = torch.flip(mask1,[2,3])
        mask = mask1 * mask2
        mask = mask.view(N,C,-1,W,H).sum(dim=2)

        m1 = mask.data.new(mask.shape).fill_(0)
        m1[mask>=1.] = 1. # at least 1 dir
        m1[m1==0.] = 0.
        x_thin1 = x * m1

        # m2 = mask.data.new(mask.shape).fill_(0)
        # m2[mask>=3.] = 1. # at least 2 dirs
        # m2[m2==0.] = 0.
        # x_thin2 = x_thin1 * m2
        
        return x_thin1
    
    def denoise(self,x):
        # N,C,W,H = x.shape
        # x_unfold = F.unfold(x, 3, 1, 1).view(N,C,3,3,W,H)
        tmp = F.conv2d(x, self.blob.cuda(), padding=1)
        mask = tmp.data.new(tmp.shape).fill_(0)
        mask[tmp>1.] = 1.
        return x * mask
    def directed_nms(self,x):
        tmp = F.conv2d(x, self.dir_filt, padding=1)

        N,C,W,H = x.shape
        x_unfold = F.unfold(x, 3, 1, 1).view(N,C,3,3,W,H)
        # 2. check every point whether it's a peak on any direction? gen a mask, 1 if satisfy.
        x_diff = x_unfold - x[:,:,None,None]
        mask1 = x_diff.data.new(x_diff.shape).fill_(0)
        mask1[x_diff<0.] = 1.
        mask2 = torch.flip(mask1,[2,3])
        mask = mask1 * mask2

        # import pdb;pdb.set_trace()
        mask = mask.view(9,W,H)
        inds = tmp.max(dim=1)[1]
        mask = torch.gather(mask,dim=0,index=inds).unsqueeze(0)

        return x * mask

    def thin_edge(self,x):
        # G, D = self.filt(x)
        # x_thin = self.nms(x, D)
        # 1. unfold
        # import pdb;pdb.set_trace()
        x_nms = self.nms(x)
        x_thin = self.denoise(x_nms)
        # x_thin = self.directed_nms(x_thin)
        return x_thin
        
class EdgeDisplacement(Net):

    def __init__(self, crf_conf):
        super(EdgeDisplacement, self).__init__(crf_conf)
        self.sobel = Sobel()

    def forward(self, x, unary, label, out_settings=None):
        def flip_add(inp,keepdim=True):
            return inp[0:1]+inp[1:2].flip(-1)
        # import pdb;pdb.set_trace()
        
        unary = self.make_unary_for_infer(unary, label.clone())

        x1 = x[0].squeeze()
        clsbd = self.infer_clsbd(x1)[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd = torch.sigmoid(flip_add(clsbd)/2)

        x2 = x[1].squeeze()
        clsbd2 = self.infer_clsbd(x2)
        clsbd2 = F.interpolate(clsbd2,scale_factor=1/1.48,mode='bilinear',align_corners=True)[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd2 = torch.sigmoid(flip_add(clsbd2)/2)

        x3 = x[2].squeeze()
        clsbd3 = self.infer_clsbd(x3)
        clsbd3 = F.interpolate(clsbd3,scale_factor=2.,mode='bilinear',align_corners=True)[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd3 = torch.sigmoid(flip_add(clsbd3)/2)

        clsbd = (clsbd+clsbd2+clsbd3)/3
        clsbd = self.sobel.thin_edge(clsbd)
        unary_ret = unary.clone()
        # clsbd.fill_(0.)
        pred = self.convcrf(unary, clsbd, label, num_iter=100)
        return pred, clsbd, unary_ret



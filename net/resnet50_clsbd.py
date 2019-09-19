import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50
from net.modules import ClsbdCRF

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
    'filter_size': 9,
    'blur': 2,
    'merge': False,
    'norm': 'none',
    'weight': 'vector',
    "unary_weight": 1.,
    "weight_init": 0.08,
    "pos_weight":10.,
    "neg_weight":1.,

    'trainable': False,
    'convcomp': True,
    'logsoftmax': True,  # use logsoftmax for numerical stability
    'softmax': True,
    'final_softmax': False,

    'pos_feats': {
        'sdims': 30,
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


class Net(nn.Module):

    def __init__(self, cam_net, crf_conf):
        super(Net, self).__init__()
        self.cam_net = cam_net
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

    def make_unary_for_infer(self, unary_raw, label):
        # unary_raw [N, 21, W, H]
        # 1. rescale
        unary_raw = F.interpolate(unary_raw, label.shape[-2:], mode='bilinear', align_corners=False)#[0] #torch.unsqueeze(unary_raw, 0)
        # 2. add background
        unary_raw = F.pad(unary_raw, (0, 0, 0, 0, 1, 1, 0, 0), mode='constant',value=0.)
        # 3. create and apply mask
        label[label==255.] = 21
        label = label.unsqueeze(1)
        mask = torch.zeros_like(unary_raw).cuda()
        mask = mask.scatter_(1,label.type(torch.cuda.LongTensor),1.)
        unary = (unary_raw * mask)[:,:-1]
        # unary[unary>0.] = 150.
        # tmp = mask[:,:-1].sum(dim=2,keepdim=True).sum(dim=3,keepdim=True) # [N,21,1,1]
        # tmp[tmp>0.] = 1.
        # unary += tmp
        # unary[:,0] = 1.
        return unary 

    def make_unary_for_train(self, unary_raw, label):
        # unary_raw [N, 21, W, H]
        # 1. rescale
        unary_raw = F.interpolate(unary_raw, label.shape[-2:], mode='bilinear', align_corners=False)#[0] #torch.unsqueeze(unary_raw, 0)
        # 2. add background
        unary_raw = F.pad(unary_raw, (0, 0, 0, 0, 1, 1, 0, 0), mode='constant',value=1.)
        # 3. create and apply mask
        label[label==255.] = 21
        label = label.unsqueeze(1)
        mask = torch.zeros_like(unary_raw).cuda()
        mask = mask.scatter_(1,label.type(torch.cuda.LongTensor),1.)
        unary = (unary_raw * mask)[:,:-1]
        unary[unary>0.] = 100.
        return unary 

    def forward(self, x, label):
        unary_raw = self.cam_net(x)
        unary_raw = F.relu(unary_raw).detach()
        unary = self.make_unary_for_train(unary_raw, label)
        clsbd = self.infer_clsbd(x)[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd = torch.sigmoid(clsbd)
        pred = self.convcrf(unary, clsbd, label, num_iter=1)
        hms = self.save_hm(unary,clsbd.repeat(1,21,1,1),pred)
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


class EdgeDisplacement(Net):

    def __init__(self, cam, crf_conf):
        super(EdgeDisplacement, self).__init__(cam, crf_conf)

    def forward(self, x, label, out_settings=None):
        # edge_out, _ = super().forward(x, label)
        def flip_add(inp,keepdim=True):
            return inp[0:1]+inp[1:2].flip(-1)
        # import pdb;pdb.set_trace()
        
        x1 = x[0].squeeze()
        unary_raw = self.cam_net(x1)
        unary_raw = flip_add(F.relu(unary_raw)).detach()
        unary = self.make_unary_for_infer(unary_raw, label.clone())
        clsbd = self.infer_clsbd(x1)[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd = torch.sigmoid(flip_add(clsbd)/2)
        
        x2 = x[1].squeeze()
        unary_raw2 = self.cam_net(x2)
        unary_raw2 = flip_add(F.relu(unary_raw2)).detach()
        unary_raw2 = F.interpolate(unary_raw2,scale_factor=2,mode='bilinear',align_corners=False)
        unary2 = self.make_unary_for_infer(unary_raw2, label.clone())
        clsbd2 = self.infer_clsbd(x2)#[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd2 = F.interpolate(clsbd2,scale_factor=2,mode='bilinear',align_corners=False)[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd2 = torch.sigmoid(flip_add(clsbd2)/2)
        
        # clsbd *= clsbd.max(dim=)
        pred = self.convcrf((unary+unary2)/2., (clsbd+clsbd2)/2, label, num_iter=20)
        # hms = self.save_hm(unary,clsbd.repeat(1,21,1,1),pred)
        return pred#, hms

        # edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)

        # return edge_out



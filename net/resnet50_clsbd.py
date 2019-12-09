import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50
from net.modules import ClsbdCRF
from misc.imutils import Sobel

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
        'sdims': 100,
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
infer_conf2 = {
    'filter_size': 5,
    'blur': 2,
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


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])

class Net(nn.Module):

    def __init__(self):
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

        self.convcrf = ClsbdCRF(default_conf, nclasses=21)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.last_two = nn.ModuleList([self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])
        self.sobel = Sobel()

    def infer_clsbd(self, x): # no sigmoid
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1)#.detach()
        x3 = self.stage3(x2)#.detach()
        x4 = self.stage4(x3)#.detach()
        x5 = self.stage5(x4)#.detach()

        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        edge_up = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        edge_out = edge_up
        return edge_out

    def _forward(self, img, unary, num_iter=1, mask=None):
        # NOTE: assume unary (label) is well prepared as a tensor.
        # unary_raw = self.cam_net(x)
        # unary_raw = F.relu(unary_raw).detach()
        # if self.training:
        #     unary = self.make_unary_for_train(label)
        # else:
        #     unary = self.make_unary_for_infer(label)
        clsbd = self.infer_clsbd(img)[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd = torch.sigmoid(clsbd)
        pred = self.convcrf(unary, clsbd, num_iter=num_iter, mask=mask)
        hms = self.save_hm(unary,clsbd.repeat(1,21,1,1))
        return pred, hms

    def forward(self, img, unary, num_iter=1, mask=None, MSF=False):
        if not MSF:
            return self._forward(img, unary, num_iter=num_iter, mask=mask)
        else:
            return self.forwardMSF(img, unary, num_iter=num_iter, mask=mask)

    def infer_crf(self,clsbd, unary, num_iter=1, mask=None):
        clsbd = clsbd[...,:unary.shape[-2],:unary.shape[-1]]
        # if not self.training:
        #     clsbd = self.sobel.thin_edge(clsbd)
        pred = self.convcrf(unary, clsbd, num_iter=num_iter, mask=mask)
        hms = [unary,clsbd.repeat(1,21,1,1)]
        if not self.training:
            hms.append(pred)
        hms = self.save_hm(*hms)
        return pred, hms

    def forwardMSF(self,img_pack, unary, num_iter=100, mask=None):
        def flip_add(inp):
            return (inp[:,0]+inp[:,1].flip(-1))/2
        def fiveD_forward(inp):
            N = inp.shape[0]
            out = self.infer_clsbd(inp.view(N*2,*(inp.shape[2:])))
            out = out.view(N,2,*(out.shape[1:]))
            return out
        num_scales = len(img_pack)
        clsbd_list = [flip_add(fiveD_forward(img)) for img in img_pack]
        std_size = clsbd_list[0].shape[-2:]
        clsbd = torch.mean(torch.stack(
            [torch.sigmoid(F.interpolate(o, std_size, mode='bilinear', align_corners=False)) for o
                in clsbd_list]), 0)
        pred, hms = self.infer_crf(clsbd, unary, num_iter=num_iter, mask=mask)
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

        return (list(self.edge_layers.parameters()), list(self.last_two.parameters()))#, tuple(self.dp_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.convcrf = ClsbdCRF(default_conf, nclasses=21).cuda()
        self.convcrf.train()
        return self

    def eval(self,mode=True):
        super().eval()
        self.convcrf1 = ClsbdCRF(infer_conf, nclasses=21).cuda()
        self.convcrf1.eval()
        self.convcrf2 = ClsbdCRF(infer_conf2, nclasses=21).cuda()
        self.convcrf2.eval()
        self.convcrf = self.convcrf1
        return self
    def usecrf(self,ind):
        if ind==1:
            self.convcrf = self.convcrf1
        else:
            self.convcrf = self.convcrf2

class EdgeDisplacement(Net):

    def __init__(self, crf_conf):
        super(EdgeDisplacement, self).__init__(crf_conf)

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
        clsbd2 = F.interpolate(clsbd2,scale_factor=2.,mode='bilinear',align_corners=True)[...,:unary.shape[-2],:unary.shape[-1]]
        clsbd2 = torch.sigmoid(flip_add(clsbd2)/2)

        clsbd = (clsbd+clsbd2)/2
        pred = self.convcrf(unary, clsbd, label, num_iter=30)
        return pred, clsbd



import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50
from net.modules import ClsbdCRF

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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

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

    def forward(self, x, unary):

        clsbd = self.infer_clsbd(x)
        clsbd = torch.sigmoid(clsbd)

        pred = self.convcrf(unary, clsbd)

        return pred

    def trainable_parameters(self):

        return (tuple(self.edge_layers.parameters())#, tuple(self.dp_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


class EdgeDisplacement(Net):

    def __init__(self):
        super(EdgeDisplacement, self).__init__()

    def forward(self, x, out_settings=None):
        edge_out = super().forward(x)

        edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)

        return edge_out



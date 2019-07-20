import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50

from .modules import Gap, KQ, Relation

KQ_FT_DIM = 32
KQ_DIM = 16

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)

        self.branch_rel = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),
        )

        self.n_class = 20
        self.kq = KQ(1024, KQ_DIM)
        
        self.gap = Gap(2048, self.n_class)
        self.relation = Relation(self.n_class, KQ_DIM, self.n_class, n_heads=1, rel_pattern=[(3,3),(3,2),(5,2)]) #,(5,5)
        
        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.newly_added = nn.ModuleList([self.branch_rel, self.kq, self.relation, self.gap])

    def forward(self, x):

        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3)
        feats_loc = self.stage5(x4)  # N, 2048, KQ_FT_DIM, KQ_FT_DIM
        feats_rel = self.branch_rel(x4)

        K, Q = self.kq(feats_rel)
        pred0, cam0 = self.gap(feats_loc)
        pred1, cam1 = self.relation(cam0, K, Q)

        hms = self.save_hm(cam0, cam1)
        
        return [pred1], pred0, hms

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

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x, label):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3)
        feats_loc = self.stage5(x4)  # N, 2048, KQ_FT_DIM, KQ_FT_DIM
        feats_rel = self.branch_rel(x4)

        K, Q = self.kq(feats_rel)
        pred0, cam0 = self.gap(feats_loc)
        pred1, cam1 = self.relation(cam0, K, Q)

        # x = F.conv2d(x, self.classifier.weight)
        x = F.relu(cam1)
        
        x = x[0] + x[1].flip(-1)

        return x

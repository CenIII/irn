import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50

from .modules import Gap, KQ, Bayes

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

        self.fc_kq_ft1 = nn.Sequential(
            nn.Conv2d(64, KQ_FT_DIM, 1, bias=False),
            nn.GroupNorm(4, KQ_FT_DIM),
            nn.MaxPool2d(4),
            nn.ReLU(inplace=True),
        )
        self.fc_kq_ft2 = nn.Sequential(
            nn.Conv2d(256, KQ_FT_DIM, 1, bias=False),
            nn.GroupNorm(4, KQ_FT_DIM),
            nn.MaxPool2d(4),
            nn.ReLU(inplace=True),
        )
        self.fc_kq_ft3 = nn.Sequential(
            nn.Conv2d(512, KQ_FT_DIM, 1, bias=False),
            nn.GroupNorm(4, KQ_FT_DIM),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )
        self.fc_kq_ft4 = nn.Sequential(
            nn.Conv2d(1024, KQ_FT_DIM, 1, bias=False),
            nn.GroupNorm(4, KQ_FT_DIM),
            # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_kq_ft5 = nn.Sequential(
            nn.Conv2d(2048, KQ_FT_DIM, 1, bias=False),
            nn.GroupNorm(4, KQ_FT_DIM),
            # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )

        self.n_class = 20
        self.kq = KQ(5*KQ_FT_DIM, KQ_DIM)
        self.feature = nn.Conv2d(2048,64,1)
        self.bayes = Bayes(64, KQ_DIM, n_class=self.n_class, n_heads=1, dif_pattern=[(3, 5)], rel_pattern=[(5, 3)])
        self.gap = Gap(64, self.n_class)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.fc_kq_ft1, self.fc_kq_ft2, self.fc_kq_ft3, self.fc_kq_ft4, self.fc_kq_ft5])
        self.newly_added = nn.ModuleList([self.feature, self.kq,self.bayes,self.gap])

    def forward(self, x, label):

        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)  # N, 2048, KQ_FT_DIM, KQ_FT_DIM

        kq_ft1 = self.fc_kq_ft1(x1)
        kq_ft2 = self.fc_kq_ft2(x2)
        kq_ft3 = self.fc_kq_ft3(x3)[..., :kq_ft2.size(2), :kq_ft2.size(3)]
        kq_ft4 = self.fc_kq_ft4(x4)[..., :kq_ft2.size(2), :kq_ft2.size(3)]
        kq_ft5 = self.fc_kq_ft5(x5)[..., :kq_ft2.size(2), :kq_ft2.size(3)]
        kq_ft_up = torch.cat([kq_ft1, kq_ft2, kq_ft3, kq_ft4, kq_ft5], dim=1)

        K,Q = self.kq(kq_ft_up)
        feats = self.feature(x5)[..., :kq_ft2.size(2), :kq_ft2.size(3)]
        feats1, preds1 = self.bayes(feats, K, Q, label)
        pred = self.gap(feats1, save_hm=True)

        preds1.append(pred)

        return preds1

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
        x5 = self.stage5(x4)  # N, 2048, KQ_FT_DIM, KQ_FT_DIM

        kq_ft1 = self.fc_kq_ft1(x1)
        kq_ft2 = self.fc_kq_ft2(x2)
        kq_ft3 = self.fc_kq_ft3(x3)[..., :kq_ft2.size(2), :kq_ft2.size(3)]
        kq_ft4 = self.fc_kq_ft4(x4)[..., :kq_ft2.size(2), :kq_ft2.size(3)]
        kq_ft5 = self.fc_kq_ft5(x5)[..., :kq_ft2.size(2), :kq_ft2.size(3)]
        kq_ft_up = torch.cat([kq_ft1, kq_ft2, kq_ft3, kq_ft4, kq_ft5], dim=1)

        K,Q = self.kq(kq_ft_up)
        feats = self.feature(x5)[..., :kq_ft2.size(2), :kq_ft2.size(3)]
        feats1, preds1 = self.bayes(feats, K, Q, label)
        pred = self.gap(feats1, save_hm=True)

        # x = F.conv2d(x, self.classifier.weight)
        x = F.relu(self.gap.heatmaps.permute(0,3,1,2))
        
        x = x[0] + x[1].flip(-1)

        return x

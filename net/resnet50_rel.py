import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50

from .modules import Gap, KQ, Relation

KQ_FT_DIM = 32
KQ_DIM = 32

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)

        # self.branch_rel = nn.Sequential(
        #     nn.Conv2d(1024, 1024, 1),
        # )
        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            # nn.Conv2d(64, 64, 1, bias=False),
            # nn.GroupNorm(4, 32),
            nn.MaxPool2d(2),
            # nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            # nn.Conv2d(256, 128, 1, bias=False),
            # nn.GroupNorm(4, 32),
            nn.MaxPool2d(2)
            # nn.ReLU(inplace=True),
        )
        # self.fc_edge3 = nn.Sequential(
        #     # nn.Conv2d(512, 256, 1, bias=False),
        #     # nn.GroupNorm(4, 32),
        #     nn.MaxPool2d(2)
        #     # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     # nn.ReLU(inplace=True),
        # )
        self.fc_edge4 = nn.Sequential(
            # nn.Conv2d(1024, 512, 1, bias=False),
            # nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.ReLU(inplace=True),
        )

        self.n_class = 20
        self.gap = Gap(2048, self.n_class)

        # self.high_kq = KQ(512+1024, KQ_DIM) # 32
        # self.high_rel = Relation(self.n_class, KQ_DIM, self.n_class, n_heads=1, rel_pattern=[(5,6),(5,4)])  # 2,0,0,1,0,1,1,0,0,0,1

        self.low_kq = KQ(64+256, KQ_DIM) # 64
        self.low_rel = Relation(self.n_class, KQ_DIM, self.n_class, 
                                n_heads=1, rel_pattern=[(3,2),(5,1),(5,3),(5,5)]) #,(5,5)

        self.upscale_cam = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.backbone = nn.ModuleList([self.stage4, self.stage5]) #self.stage1, self.stage2, self.stage3, 
        self.convs = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge4, self.low_kq])
        self.leaf_gaps = nn.ModuleList([self.gap])

    def forward(self, x):

        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3)
        feats_loc = self.stage5(x4)  # N, 2048, KQ_FT_DIM, KQ_FT_DIM

        edge1 = self.fc_edge1(x1) # 64
        edge2 = self.fc_edge2(x2) # 64
        edge3 = x3 # 32
        edge4 = self.fc_edge4(x4) # 32
        feats_low_rel = torch.cat([edge1, edge2], dim=1)

        pred0, cam0 = self.gap(feats_loc)

        cam0 = self.upscale_cam(cam0)[..., :edge2.size(2), :edge2.size(3)]

        Kl, Ql = self.low_kq(feats_low_rel)
        # pred1, cam1 = self.high_rel(cam0, Kl, Ql)
        pred2, cam2 = self.low_rel(cam0, Kl, Ql)
        pred3, cam3 = self.low_rel(cam2, Kl, Ql)

        hms = self.save_hm(cam0, cam2, cam3)
        
        return [pred2, pred3], pred0, hms  #pred1, 

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
        return (list(self.backbone.parameters()), list(self.convs.parameters()), list(self.leaf_gaps.parameters())) #(list(self.backbone.parameters()), 


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3)
        feats_loc = self.stage5(x4)  # N, 2048, KQ_FT_DIM, KQ_FT_DIM

        edge1 = self.fc_edge1(x1) # 64
        edge2 = self.fc_edge2(x2) # 64
        edge3 = self.fc_edge3(x3) # 32
        edge4 = x4 # 32
        feats_low_rel = torch.cat([edge1, edge2], dim=1)
        feats_high_rel = torch.cat([edge3, edge4], dim=1)

        pred0, cam0 = self.gap(feats_loc)
        Kh, Qh = self.high_kq(feats_high_rel)
        pred1, cam1 = self.high_rel(cam0, Kh, Qh)

        cam1 = self.upscale_cam(cam1)[..., :edge2.size(2), :edge2.size(3)]

        Kl, Ql = self.low_kq(feats_low_rel)
        pred2, cam2 = self.low_rel(cam1, Kl, Ql)
        pred3, cam3 = self.low_rel(cam2, Kl, Ql)
        # x = F.conv2d(x, self.classifier.weight)
        x = F.relu(cam3)
        
        x = x[0] + x[1].flip(-1)

        return x

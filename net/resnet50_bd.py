import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50
from .modules import Gap, BDInfusion

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
            nn.MaxPool2d(2),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.MaxPool2d(2),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)
        
        self.nclass = 21
        self.gap0 = Gap(2048, self.nclass)
        self.upscale_cam = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.infuse = BDInfusion(self.nclass, self.nclass, rel_pattern=[(3,3),(3,6),(5,1),(5,5)])
        

        self.backbone = nn.ModuleList([self.stage4, self.stage5])
        self.boundary_branch = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6, self.infuse])
        self.cam_branch = nn.ModuleList([self.gap0, self.upscale_cam])

    def forward(self, x):

        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        boundary = torch.sigmoid(self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1)))
        
        pred0, cam0 = self.gap0(x5)
        cam0 = self.upscale_cam(cam0)[..., :edge2.size(2), :edge2.size(3)]
        pred1, cam1 = self.infuse(cam0, boundary)
        pred2, cam2 = self.infuse(cam1, boundary)

        hms = self.save_hm(cam0, boundary, cam2)
        
        return [pred1, pred2], pred0, hms

    def getHeatmaps(self, hms, classid):
        hm = []
        for heatmap in hms:
            if len(heatmap.squeeze().shape) == 3:
                hm.append(heatmap.squeeze())
                continue
            zzz = classid[:, None, None, None].repeat(1, heatmap.shape[1], heatmap.shape[2], 1)
            hm.append(torch.gather(heatmap, 3, zzz).squeeze())
        return hm

    def save_hm(self, *cams):
        hm = []
        for cam in cams:
            if len(cam.squeeze().shape) == 3:
                hm.append(cam)
                continue
            hm.append(cam.permute(0,2,3,1))
        return hm

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.boundary_branch.parameters()), list(self.cam_branch.parameters()))



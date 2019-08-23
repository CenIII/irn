import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50

from .modules import Gap, KQ, Relation

KQ_DIM = 8

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
        # self.kq = KQ(64+256+512+1024, KQ_DIM) #512+1024
        
        self.gap = Gap(2048, self.n_class)
        self.upscale_cam = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.relation = Relation(self.n_class, KQ_DIM, self.n_class, n_heads=1, 
        #                         rel_pattern=[(3,2),(5,1),(5,3),(5,5)]) 
        self.bgap = Gap(2048, self.n_class)
        self.backbone = nn.ModuleList([self.stage4, self.stage5]) #self.stage1, self.stage2, self.stage3, 
        self.convs = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge4]) #, self.kq
        self.leaf_gaps = nn.ModuleList([self.gap, self.bgap])

    def infer(self, x, mask, train=True):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3)
        feats_loc = self.stage5(x4)  # N, 2048, KQ_FT_DIM, KQ_FT_DIM

        # edge1 = self.fc_edge1(x1)
        # edge2 = self.fc_edge2(x2)
        # edge3 = x3[..., :edge2.size(2), :edge2.size(3)]
        # edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        # feats_rel = torch.cat([edge1, edge2, edge3, edge4], dim=1) #edge1, edge2, 

        # K, Q = self.kq(feats_rel)
        pred0, cam0 = self.gap(feats_loc)
        pred1, cam1 = self.bgap(F.normalize(feats_loc.detach(),dim=1)*10,mask=mask)
        # if train:
        #     K_d, Q_d = F.max_pool2d(K,2), F.max_pool2d(Q,2)
        # else:
        #     K_d, Q_d = F.max_pool2d(K,2,padding=1)[..., :cam0.size(2), :cam0.size(3)], F.max_pool2d(Q,2,padding=1)[..., :cam0.size(2), :cam0.size(3)]
        # pred1, cam1 = self.relation(cam0, K_d, Q_d)
        # cam1 = self.upscale_cam(cam1)[..., :edge2.size(2), :edge2.size(3)]
        # pred2, cam2 = self.relation(cam1, K, Q)
        # ftnorm = torch.norm(feats_loc,dim=1).detach().data
        return pred0, cam0, [pred1], [cam1]#, ftnorm 

    def make_class_boundary(normed_cam, label):
        # TODO: implement this. 
        import pdb;pdb.set_trace()
        pass
        return clsbd_list

    def forward(self, x, mask, label):

        pred0, cam0, preds, cams = self.infer(x, mask)
        clsbd_list = self.make_class_boundary(cams[-1],label)
        hms = self.save_hm(cam0, cams[0])
        hms.append(clsbd_list[0])
        
        return preds, pred0, hms, clsbd_list

    def getHeatmaps(self, hms, classid):
        hm = []
        for heatmap in hms:
            if heatmap.dim()<4:
                hm.append(heatmap)
                continue
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
        return (list(self.backbone.parameters()), list(self.convs.parameters()), list(self.leaf_gaps.parameters()))


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):
        pred0, cam0, preds, cams = self.infer(x, train=False)
        # x = F.conv2d(x, self.classifier.weight)
        x = F.relu(cams[-1])
        
        x = x[0] + x[1].flip(-1)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50

from .modules import Gap, KQ, Relation
import copy 

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
        self.stage4b = nn.Sequential(copy.deepcopy(self.resnet50.layer3))
        self.stage5b = nn.Sequential(copy.deepcopy(self.resnet50.layer4))

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
        self.backbone = nn.ModuleList([self.stage4, self.stage5, self.stage4b, self.stage5b]) #self.stage1, self.stage2, self.stage3, 
        self.convs = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge4]) #, self.kq
        self.leaf_gaps = nn.ModuleList([self.gap, self.bgap])
    
    def get_gap_weights(self):
        return self.gap.lin.weight.squeeze().detach()#F.normalize(,dim=1)

    def gather_maps(self, cam, wts, label): #[2, 20, 32, 32]
        # TODO: implement this. 
        N,C,W,H = cam.shape
        # import pdb;pdb.set_trace()
        top4 = torch.topk(wts,5,dim=1)[1]
        tmp = wts.scatter(1,top4,1.)
        tmp[tmp<1.] = 0.  #[20, 20]
        gcam = torch.matmul(F.relu(cam.view(N,C,-1).permute(0,2,1)), tmp.transpose(0,1)).permute(0,2,1)#[2,20,1024].view(N,C,W,H)
        gcam = gcam * label[...,None]
        # classify on pixel.
        zzz = torch.max(gcam,dim=1) 
        mask = torch.zeros_like(gcam).cuda()
        mask = mask.scatter(1,zzz[1].unsqueeze(1),1.)
        gcam = gcam * mask
        gcam = gcam.view(N,C,W,H)
        
        return gcam

    def infer(self, x, train=True):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3)
        feats_loc = self.stage5(x4)  # N, 2048, KQ_FT_DIM, KQ_FT_DIM
        x4b = self.stage4b(x3)
        feats_loc_b = self.stage5b(x4b)  # N, 2048, KQ_FT_DIM, KQ_FT_DIM

        # edge1 = self.fc_edge1(x1)
        # edge2 = self.fc_edge2(x2)
        # edge3 = x3[..., :edge2.size(2), :edge2.size(3)]
        # edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        # feats_rel = torch.cat([edge1, edge2, edge3, edge4], dim=1) #edge1, edge2, 

        # K, Q = self.kq(feats_rel)

        pred0, cam0 = self.gap(feats_loc)
        pred1, cam1 = self.bgap(feats_loc_b) # F.normalize(feats_loc.detach(),dim=1)*10,mask=mask)
        # if train:
        #     K_d, Q_d = F.max_pool2d(K,2), F.max_pool2d(Q,2)
        # else:
        #     K_d, Q_d = F.max_pool2d(K,2,padding=1)[..., :cam0.size(2), :cam0.size(3)], F.max_pool2d(Q,2,padding=1)[..., :cam0.size(2), :cam0.size(3)]
        # pred1, cam1 = self.relation(cam0, K_d, Q_d)
        # cam1 = self.upscale_cam(cam1)[..., :edge2.size(2), :edge2.size(3)]
        # pred2, cam2 = self.relation(cam1, K, Q)

        return pred0, cam0, [pred1], [cam1]

    def forward(self, x):

        pred0, cam0, preds, cams = self.infer(x)
        
        hms = self.save_hm(cam0, cams[0])
        
        return preds, pred0, hms

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
        return (list(self.backbone.parameters()), list(self.convs.parameters()), list(self.leaf_gaps.parameters()))


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x): #, wts,label
        pred0, cam0, preds, cams = self.infer(x, train=False)
        # x = F.conv2d(x, self.classifier.weight)
        
        # x = self.gather_maps(x,wts,label)
        x = F.relu(cams[-1])
        x = x[0] + x[1].flip(-1)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils, imutils
from net import resnet50

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

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)#_ASPP(2048, 21, [6, 12, 18, 24]) #

        # self.backbone_half = nn.ModuleList([self.stage3, self.stage4])
        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def _forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)#.detach()

        x = self.stage3(x)
        x = self.stage4(x)  # N, 2048, 32, 32
        
        # x = torchutils.gap2d(x, keepdims=True) # N, 2048, 1, 1
        x = self.classifier(x) # N, 20, 32, 32
        
        # x = x.view(-1, 21) # N, 20

        return x

    def forward(self, x, MSF=False):
        if not MSF:
            return self._forward(x)
        else:
            return self.forwardMSF(x)

    def forwardMSF(self, x_pack): # x_pack[0] [16, 2, 3, 512, 512]
        def flip_add(inp):
            return (inp[:,0]+inp[:,1].flip(-1))/2
        def fiveD_forward(inp):
            N = inp.shape[0]
            out = self.forward(inp.view(N*2,*(inp.shape[2:])))
            out = out.view(N,2,*(out.shape[1:]))
            return out
        # size = x_pack[0].shape[-2:]
        # strided_size = imutils.get_strided_size(size, 16)
        outputs = [flip_add(fiveD_forward(img))
                       for img in x_pack]
        strided_size = outputs[0].shape[-2:]
        strided_cam = torch.sum(torch.stack(
            [F.interpolate(o, strided_size, mode='bilinear', align_corners=False) for o
                in outputs]), 0)
        return strided_cam

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
        return self

    def init_trainable_parameters(self):
        return (list(self.backbone[2:].parameters()), list(self.newly_added.parameters()))
    
    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        # x = F.conv2d(x, self.classifier.weight)
        x = self.classifier(x)
        x = F.relu(x)
        # x = torchutils.leaky_log(x,leaky_rate=0.) #torch.log(1+F.relu(x))
        
        x = x[0] + x[1].flip(-1)

        return x

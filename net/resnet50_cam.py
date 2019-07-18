import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)
        self.saliency_classifier = nn.Conv2d(2048, 1, 1, bias=False)
        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.saliency_classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)  # N, 2048, 32, 32

        x = torchutils.gap2d(x, keepdims=True) # N, 2048, 1, 1
        sal = self.saliency_classifier(x.detach()) # N, 1,1,1
        x = self.classifier(x) # N, 20, 1, 1
        
        x = x.view(-1, 20) # N, 20
        sal = sal.view(-1, 1) # N, 1
        return x, sal

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

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        sal = F.conv2d(x, self.saliency_classifier.weight)
        sal = F.relu(sal)
        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)
        
        sal = sal[0] + sal[1].flip(-1)
        x = x[0] + x[1].flip(-1)

        return x, sal

import time, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PPM(nn.Module):
    def __init__(self):
        super(PPM, self).__init__()
        stages = [1, 2, 3, 6]
        self.features = []
        for s in stages:
            self.features.append(nn.Sequential(
                nn.AvgPool2d(s),
                nn.Conv2d(2048, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256, momentum=.95),
                nn.ReLU(inplace=True)
            ))

        self.features = nn.ModuleList(self.features)

    def forward(self, feature):
        feature_size = feature.size()
        out = [feature]
        for f in self.features:
            out.append(F.upsample(f(feature), feature_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)

        return out


class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
        self.name = time.asctime() + " PSPnet "

        resnet = models.resnet101(pretrained=True)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = PPM()

        self.final = nn.Sequential(
            nn.Conv2d(3072, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, 20, kernel_size=1)
        )

    def forward(self, image):
        torch.cuda.empty_cache()

        imagr_size = image.size()

        image = self.downsample(image)
        image = self.downsample(image)

        feature = self.layer0(image)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)

        feature = self.ppm(feature)
        # feature = F.upsample(feature, imagr_size[2:], mode='bilinear')

        out = self.final(feature)

        out = F.upsample(out, imagr_size[2:], mode='bilinear')
        return out

    def load_weights(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))

    def save_weights(self, path=''):
        torch.save(self.state_dict(), os.path.join('weights/' + self.name + path))

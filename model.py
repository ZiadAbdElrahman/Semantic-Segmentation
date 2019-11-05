import torch.nn as nn
import torchvision.models as models
import torch, torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.Resnet = models.resnet50(pretrained=True)
        self.Resnet = nn.Sequential(
            *[self.Resnet.conv1, self.Resnet.layer1, self.Resnet.layer2, self.Resnet.layer3, self.Resnet.layer4])

    def forward(self, image):
        feature = self.Resnet(image)
        return feature


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.Stage1Pool = nn.AdaptiveAvgPool2d(1)
        self.Stage2Pool = nn.AdaptiveAvgPool2d(2)
        self.Stage3Pool = nn.AdaptiveAvgPool2d(3)
        self.Stage4Pool = nn.AdaptiveAvgPool2d(6)

        self.BNstage1 = nn.BatchNorm2d(64)
        self.BNstage2 = nn.BatchNorm2d(64)
        self.BNstage3 = nn.BatchNorm2d(64)
        self.BNstage4 = nn.BatchNorm2d(64)

        self.Stage1Conv = nn.Conv2d(2048, 64, kernel_size=1)
        self.Stage2Conv = nn.Conv2d(2048, 64, kernel_size=1)
        self.Stage3Conv = nn.Conv2d(2048, 64, kernel_size=1)
        self.Stage4Conv = nn.Conv2d(2048, 64, kernel_size=1)

        self.conv1 = nn.ConvTranspose2d(in_channels=2304, out_channels=512, kernel_size=(2, 2), stride=2)
        self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2)
        self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2)
        self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=29, kernel_size=(2, 2), stride=2)

        self.Relu = nn.ReLU()

    def forward(self, feature):
        s1 = self.Stage1Pool(feature)
        s1 = self.Stage1Conv(s1)
        s1 = self.BNstage1(s1)
        s1 = self.Relu(s1)

        s2 = self.Stage2Pool(feature)
        s2 = self.Stage2Conv(s2)
        s2 = self.BNstage1(s2)
        s2 = self.Relu(s2)

        s3 = self.Stage3Pool(feature)
        s3 = self.Stage3Conv(s3)
        s3 = self.BNstage1(s3)
        s3 = self.Relu(s3)

        s4 = self.Stage4Pool(feature)
        s4 = self.Stage4Conv(s4)
        s4 = self.BNstage1(s4)
        s4 = self.Relu(s4)

        s1 = F.interpolate(s1, size=(16, 16), mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=(16, 16), mode='bilinear', align_corners=True)
        s3 = F.interpolate(s3, size=(16, 16), mode='bilinear', align_corners=True)
        s4 = F.interpolate(s4, size=(16, 16), mode='bilinear', align_corners=True)
        conc = torch.cat((feature, s1, s2, s3, s4), 1)

        output = self.conv1(conc)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)

        return output

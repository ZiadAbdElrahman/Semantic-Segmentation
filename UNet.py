import torch.nn as nn
import torch, torch.nn.functional as F


class EcoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EcoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, 2 * out_ch, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(2 * out_ch, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * out_ch, 2 * out_ch, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(2 * out_ch, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * out_ch, out_ch, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.model_name = "Unet"

        self.EB1 = EcoderBlock(3, 64)
        self.EB2 = EcoderBlock(64, 128)
        self.EB3 = EcoderBlock(128, 256)
        self.EB4 = EcoderBlock(256, 512)

        self.center = DecoderBlock(512, 512)

        self.DB4 = DecoderBlock(1024, 256)
        self.DB3 = DecoderBlock(512, 128)
        self.DB2 = DecoderBlock(256, 64)
        self.DB1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(64, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(64, momentum=0.95),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, 29, kernel_size=1)

    def forward(self, x):
        EB1_out = self.EB1(x)
        EB2_out = self.EB2(EB1_out)
        EB3_out = self.EB3(EB2_out)
        EB4_out = self.EB4(EB3_out)

        center_out = self.center(EB4_out)

        DB4 = self.DB4(torch.cat([F.upsample(EB4_out, center_out.size()[2:], mode='bilinear'), center_out], 1))
        DB3 = self.DB3(torch.cat([F.upsample(EB3_out, DB4.size()[2:], mode='bilinear'), DB4], 1))
        DB2 = self.DB2(torch.cat([F.upsample(EB2_out, DB3.size()[2:], mode='bilinear'), DB3], 1))
        DB1 = self.DB1(torch.cat([F.upsample(EB1_out, DB2.size()[2:], mode='bilinear'), DB2], 1))

        out = self.final(DB1)
        return F.upsample(out, x.size()[2:], mode='bilinear')

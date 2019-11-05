import torch.nn as nn
import torchvision.models as models
import torch, torch.nn.functional as F


class EcoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EcoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernal_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, 2 * out_ch, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * out_ch, 2 * out_ch, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * out_ch, out_ch, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, number_of_block):
        super(UNet, self).__init__()
        self.EB1 = EcoderBlock(3, 64)
        self.EB2 = EcoderBlock(64, 128)
        self.EB3 = EcoderBlock(128, 256)
        self.EB4 = EcoderBlock(256, 512)

        self.center = nn.Sequential(
            nn.MaxPool2d(kernal_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )

        self.DB4 = DecoderBlock(1024, 256)
        self.DB3 = DecoderBlock(512, 128)
        self.DB2 = DecoderBlock(256, 64)
        self.DB1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, 29, kernal_size=1)

    def forword(self, x):
        EB1_out = self.EB1(x)
        EB2_out = self.EB2(EB1_out)
        EB3_out = self.EB2(EB2_out)
        EB4_out = self.EB2(EB3_out)

        center_out = self.center(EB4_out)

        DB4 = self.DB4(torch.concat((EB4_out, center_out), 1))
        DB3 = self.DB3(torch.concat((EB3_out, DB4), 1))
        DB2 = self.DB2(torch.concat((EB2_out, DB3), 1))
        DB1 = self.DB1(torch.concat((EB1_out, DB2), 1))

        out = self.final(DB1)
        return out

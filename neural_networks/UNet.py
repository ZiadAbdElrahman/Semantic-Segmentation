import torch, torch.nn.functional as F
import torch.nn as nn
import time, os


class EcoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EcoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.95),
            nn.ReLU(inplace=True),

        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.block(x)
        y_pooled = self.pool(y)
        return y, y_pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, int(out_ch / 2), kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.name = time.asctime() + "Unet "
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.EB1 = EcoderBlock(3, 32)
        self.EB2 = EcoderBlock(32, 64)
        self.EB3 = EcoderBlock(64, 128)
        self.EB4 = EcoderBlock(128, 256)

        self.center = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )

        self.DB4 = DecoderBlock(512, 256)
        self.DB3 = DecoderBlock(256, 128)
        self.DB2 = DecoderBlock(128, 64)
        self.DB1 = DecoderBlock(64, 64)
        self.final = nn.Conv2d(32, 20, kernel_size=1)

    def forward(self, x):
        x_ = self.downsample(x)

        EB1_out, x_ = self.EB1(x_)
        EB2_out, x_ = self.EB2(x_)
        EB3_out, x_ = self.EB3(x_)
        EB4_out, x_ = self.EB4(x_)

        center_out = self.center(x_)

        DB4 = self.DB4(torch.cat([EB4_out, center_out], 1))
        DB3 = self.DB3(torch.cat([EB3_out, DB4], 1))
        DB2 = self.DB2(torch.cat([EB2_out, DB3], 1))
        DB1 = self.DB1(torch.cat([EB1_out, DB2], 1))

        out = self.final(F.upsample(DB1, x.size()[2:], mode='bilinear'))
        return out

    def load_weights(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))

    def save_weights(self, path=''):
        torch.save(self.state_dict(), os.path.join('weights/' + self.name + path))

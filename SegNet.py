import torch.nn as nn
import torchvision.models as models
import torch, torch.nn.functional as F
import time


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ley):
        super(EncoderBlock, self).__init__()
        if ley == 2:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )
        elif ley == 3:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ley):
        super(DecoderBlock, self).__init__()
        if ley == 2:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(in_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )
        elif ley == 3:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(in_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(in_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.Encoder_Block1 = EncoderBlock(3, 32, 2)
        self.Encoder_Block2 = EncoderBlock(32, 64, 2)
        self.Encoder_Block3 = EncoderBlock(64, 128, 3)
        self.Encoder_Block4 = EncoderBlock(128, 256, 3)
        self.Encoder_Block5 = EncoderBlock(256, 256, 3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

    def forward(self, x):
        x = self.Encoder_Block1(x)
        x, ind1 = self.pool(x)
        x = self.Encoder_Block2(x)
        x, ind2 = self.pool(x)
        x = self.Encoder_Block3(x)
        x, ind3 = self.pool(x)
        x = self.Encoder_Block4(x)
        x, ind4 = self.pool(x)

        out = self.Encoder_Block5(x)

        return out, ind1, ind2, ind3, ind4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.Decoder_Block1 = DecoderBlock(32, 32, 2)
        self.Decoder_Block2 = DecoderBlock(64, 32, 2)
        self.Decoder_Block3 = DecoderBlock(128, 64, 3)
        self.Decoder_Block4 = DecoderBlock(256, 128, 3)
        self.Decoder_Block5 = DecoderBlock(256, 256, 3)

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.final = nn.Conv2d(32, 20, kernel_size=1, stride=1)

    def forward(self, x, ind1, ind2, ind3, ind4):
        x = self.Decoder_Block5(x)

        x = self.unpool(x, ind4)
        x = self.Decoder_Block4(x)

        x = self.unpool(x, ind3)
        x = self.Decoder_Block3(x)

        x = self.unpool(x, ind2)
        x = self.Decoder_Block2(x)

        x = self.unpool(x, ind1)
        x = self.Decoder_Block1(x)

        out = self.final(x)

        return out


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.name = "SegNet " + time.asctime()

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x_ = self.downsample(x)
        feature, ind1, ind2, ind3, ind4 = self.encoder(x_)
        out = self.decoder(feature, ind1, ind2, ind3, ind4)

        return F.upsample(out, x.size()[2:], mode='bilinear')

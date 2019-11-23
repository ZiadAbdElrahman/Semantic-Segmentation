import torch.nn as nn
import torchvision.models as models
import time


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)

        self.Encoder_Block1 = vgg.features[0:7]
        self.Encoder_Block2 = vgg.features[7:14]
        self.Encoder_Block3 = vgg.features[14:24]
        self.Encoder_Block4 = vgg.features[24:34]
        self.Encoder_Block5 = vgg.features[34:]

    def forward(self, x):
        out_b1 = self.Encoder_Block1(x)
        out_b2 = self.Encoder_Block2(out_b1)
        out_b3 = self.Encoder_Block3(out_b2)
        out_b4 = self.Encoder_Block4(out_b3)
        out = self.Encoder_Block5(out_b4)

        return {"b1": out_b1, "b2": out_b2, "b3": out_b3, "b4": out_b4, "b5": out}


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


class Decoder(nn.Module):
    def __init__(self, mood):
        super(Decoder, self).__init__()
        self.mood = mood
        self.Decoder_Block1 = DecoderBlock(64, 64, 2)
        self.Decoder_Block2 = DecoderBlock(128, 64, 2)
        self.Decoder_Block3 = DecoderBlock(256, 128, 3)
        self.Decoder_Block4 = DecoderBlock(512, 256, 3)
        self.Decoder_Block5 = DecoderBlock(512, 512, 3)

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.final = nn.Conv2d(64, 29, kernel_size=1, stride=1)

    def forward(self, x):
        if self.mood == 8:

            out = self.Decoder_Block5(x["b5"])
            out = self.Decoder_Block4(x["b4"] + out)
            out = self.Decoder_Block3(x["b3"] + out)
            out = self.Decoder_Block2(out)
            out = self.Decoder_Block1(out)

        elif self.mood == 16:
            out = self.Decoder_Block5(x["b5"])
            out = self.Decoder_Block4(x["b4"] + out)
            out = self.Decoder_Block3(out)
            out = self.Decoder_Block2(out)
            out = self.Decoder_Block1(out)
        else:
            out = self.Decoder_Block5(x["b5"])
            out = self.Decoder_Block4(out)
            out = self.Decoder_Block3(out)
            out = self.Decoder_Block2(out)
            out = self.Decoder_Block1(out)

        out = self.final(out)

        return out


class FCN(nn.Module):
    def __init__(self, mood):
        super(FCN, self).__init__()
        self.name = "FCN " + str(mood) + " " + time.asctime()
        self.encoder = Encoder()
        self.decoder = Decoder(mood)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

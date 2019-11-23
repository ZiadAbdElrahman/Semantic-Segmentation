import argparse
from train import Trainer
from sample import Sample
from PSPNet import PSPNet
from UNet import UNet
from SegNet import SegNet
from util import load_data, img_to_tensor
import torch
from torchvision import transforms
from torchvision.transforms import CenterCrop, FiveCrop
from torchvision.datasets import Cityscapes
from label import id_to_trainId_map_func, id_to_trainId_map_func
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()


parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
parser.add_argument('--load_weight', type=bool, default=False, help='load the weights or not')
parser.add_argument('--save_weight', type=bool, default=True, help='save the weights or not')
parser.add_argument('--predict', type=bool, default=True, help='predict random sample or not')
parser.add_argument('--numOfpredection', type=int, default=60, help='num of image to predict')

# Model parameters
parser.add_argument('--crop_size', type=int, default=512, help='crop size of the image ')
parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
parser.add_argument('--attention_size', type=int, default=512, help='dimension of attention')

# Training parameters
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=8e-4)
parser.add_argument('--reg', type=float, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    print(args)

    device = torch.device('cuda')
    # model = SegNet()
    model = PSPNet()
    # model = UNet()

    path = "/home/ziad/Documents/Semantic-Segmentation/cityScape"

    transform = transforms.Compose([
        img_to_tensor
    ])

    train = Cityscapes(path, split='train', mode='fine',
                       target_type='semantic', transforms=transform)
    val = Cityscapes(path, split='val', mode='fine',
                     target_type='semantic', transforms=transform)

    trainer = Trainer(args, device, model, train, val)
    trainer.train()

    sample = Sample(args, device, model, train, val)

    train = Cityscapes(path, split='train', mode='fine',
                       target_type='semantic')
    val = Cityscapes(path, split='val', mode='fine',
                     target_type='semantic')

    sample.random_sample(train, val)

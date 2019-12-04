import argparse
from train import Trainer
from sample import Sample
from evaluation import Evaluation
from neural_networks.PSPNet import PSPNet
from neural_networks.UNet import UNet
from neural_networks.FCN import FCN
from neural_networks.SegNet import SegNet
from util import img_to_tensor
import torch
from torchvision import transforms
from torchvision.datasets import Cityscapes

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='weights/Unet Thu Nov 28 10:54:44 2019',
                    help='path to save or load trained models in ')
parser.add_argument('--load_weight', type=bool, default=False, help='load the weights or not')
parser.add_argument('--save_weight', type=bool, default=True, help='save the weights or not')
parser.add_argument('--predict', type=bool, default=True, help='predict random sample or not')
parser.add_argument('--number_Of_predection', type=int, default=60, help='num of image to predict')
parser.add_argument('--batch_loss_print', type=int, default=100, help='print the batch size every how many steps')

# Training parameters
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--eva_batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=8e-4)
parser.add_argument('--reg', type=float, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    print(args)

    device = torch.device('cuda')
    # model = SegNet()
    model = PSPNet()
    # model = FCN(8)
    # model = UNet()

    path = "/home/ziad/Desktop/Semantic-Segmentation/cityScape"
    transform = transforms.Compose([
        img_to_tensor
    ])

    train = Cityscapes(path, split='train', mode='fine',
                       target_type='semantic', transforms=transform)
    val = Cityscapes(path, split='val', mode='fine',
                     target_type='semantic', transforms=transform)

    trainer = Trainer(args, device, model, train, val, False)
    trainer.train()

    train = Cityscapes(path, split='train', mode='fine',
                       target_type='semantic')
    val = Cityscapes(path, split='val', mode='fine',
                     target_type='semantic')

    sample = Sample(args, device, model)
    sample.random_sample(train, val)

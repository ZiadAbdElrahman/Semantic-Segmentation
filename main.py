import argparse
from train import Trainer
from sample import Sample
from model import Encoder, Decoder
from util import load_data
import torch
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
parser.add_argument('--load_weight', type=bool, default=False, help='load the weights or not')
parser.add_argument('--save_weight', type=bool, default=False, help='save the weights or not')
parser.add_argument('--predict', type=bool, default=True, help='predict random sample or not')
parser.add_argument('--numOfpredection', type=int, default=50, help='num of image to predict')

# Model parameters
parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
parser.add_argument('--attention_size', type=int, default=512, help='dimension of attention')

# Training parameters
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=6e-4)
parser.add_argument('--reg', type=float, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    print(args)

    device = torch.device('cpu')

    encoder = Encoder()
    decoder = Decoder()

    test_in, test_out = load_data(val=True)
    train_in, train_out = load_data()

    trainer = Trainer(args, device, encoder, decoder, (train_in, train_out), (test_in, test_out))
    trainer.train()

    sample = Sample(args, device, encoder, decoder, (train_in, train_out), (test_in, test_out))
    sample.random_sample()

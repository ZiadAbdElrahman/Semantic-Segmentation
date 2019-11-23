import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from util import save_weights, load_weights
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
from torchvision import transforms
from PIL import Image
from label import id_to_trainId_map_func, traindId_to_color_map_func


class Trainer:
    def __init__(self, args, device, model, training_data, val_data):
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.train_data = training_data
        self.val_data = val_data

    def train(self):
        writer = SummaryWriter()

        if self.args.load_weight:
            load_weights(self.model, self.args.model_path)

        for epoch in range(self.args.num_epochs):
            params = list(self.model.parameters())
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=params, lr=self.args.learning_rate)

            training_loss = self.step(criterion=criterion,
                                      data=self.train_data,
                                      optimizer=optimizer)

            with torch.no_grad():
                test_loss = self.step(criterion=criterion,
                                      data=self.val_data)

            writer.add_scalars('loss', {'train': training_loss,
                                        'val': test_loss}, epoch + 1)

            print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, TestLoss: {:.4f}, TestPerplexity: {:5.4f}'
                  .format(epoch + 1, self.args.num_epochs, training_loss,
                          np.exp(training_loss), test_loss, np.exp(test_loss)))

            self.args.learning_rate *= 0.995

        if self.args.save_weight:
            save_weights(self.model, self.args.model_path + self.model.name)

    def step(self, criterion, data, optimizer=None):

        data_loader = DataLoader(data, batch_size=self.args.batch_size, shuffle=False)
        total_loss = 0
        step = 0
        num_of_step = int((data_loader.__len__()) / self.args.batch_size)

        for x, y in data_loader:
            transform = transforms.Compose([
                id_to_trainId_map_func
            ])
            y = torch.from_numpy(transform(y))

            outputs = self.model(x.permute(0, 3, 1, 2).float().to(self.device))
            loss = criterion(outputs, y.long().to(self.device))

            total_loss += (loss.item() * self.args.batch_size)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (step + 1) % 100 == 0:
                print('step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(step + 1, num_of_step, loss.item(),
                              np.exp(loss.item())))

            step += 1
        return total_loss / (num_of_step * self.args.batch_size)

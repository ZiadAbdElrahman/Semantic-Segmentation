import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from label import id_to_trainId_map_func
from util import save_weights, load_weights


class Trainer:
    def __init__(self, args, device, model, training_data, val_data, loss_balance):
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.train_data = training_data
        self.val_data = val_data
        self.loss_balance = loss_balance

    def train(self):

        if self.args.load_weight:
            self.model.load_weights(self.args.model_path)

        for epoch in range(self.args.num_epochs):
            if epoch == 0:
                writer = SummaryWriter()

            params = list(self.model.parameters())
            optimizer = torch.optim.Adam(params=params, lr=self.args.learning_rate)

            training_loss = self.step(data=self.train_data,
                                      optimizer=optimizer)
            with torch.no_grad():
                test_loss = self.step(data=self.val_data)

            writer.add_scalars('loss ' + self.model.name + " " + str(self.args.learning_rate), {'train': training_loss, 'val': test_loss}, epoch + 1)

            print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, TestLoss: {:.4f}, TestPerplexity: {:5.4f}'
                  .format(epoch + 1, self.args.num_epochs, training_loss,
                          np.exp(training_loss), test_loss, np.exp(test_loss)))

            self.args.learning_rate *= 0.995

        if self.args.save_weight:
            self.model.save_weights()

    def step(self, data, optimizer=None):

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

            if self.loss_balance:

                weight = np.bincount(y.reshape(-1)) + 1
                sum_of_all = np.sum(weight)

                weight = 1 - (weight / np.sum(weight))
                weight = weight / np.sum(weight)

                weight = torch.from_numpy(weight).float().to(self.device)
                criterion = nn.CrossEntropyLoss(weight=weight)
            else:
                criterion = nn.CrossEntropyLoss()

            loss = criterion(outputs, y.long().to(self.device))
            total_loss += (loss.item() * self.args.batch_size)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (step + 1) % self.args.batch_loss_print == 0:
                print('step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(step + 1, num_of_step, loss.item(),
                              np.exp(loss.item())))
            step += 1

        return total_loss / (num_of_step * self.args.batch_size)

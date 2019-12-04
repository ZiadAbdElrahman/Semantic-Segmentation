import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from label import id_to_trainId_map_func


class Trainer:
    def __init__(self, args, device, model, training_data, val_data, loss_balance):
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.train_data = training_data
        self.val_data = val_data
        self.loss_balance = loss_balance
        # the weights is calculated with this
        #           weight = np.bincount(y.reshape(-1)) + 1
        #           weight = 1 - (weight / np.sum(weight))
        #           weight = weight / np.sum(weight)
        #           weight *= len(weight)
        self.weights = torch.tensor([0.70905324, 0.99592726, 0.83994066, 1.04652405, 1.04445648, 1.04119428,
                                     1.05069515, 1.0474944, 0.90419693, 1.04184215, 1.01517977, 1.04127241,
                                     1.05137237, 0.98745134, 1.05013918, 1.05043984, 1.05046116, 1.05171216,
                                     1.04877445, 0.93187272])

    def train(self):
        min_loss = 100
        writer = SummaryWriter()
        if self.args.load_weight:
            self.model.load_weights(self.args.model_path)
        for epoch in range(self.args.num_epochs):

            if epoch < self.args.start_epoch:
                self.args.learning_rate *= 0.995
                continue

            params = list(self.model.parameters())
            optimizer = torch.optim.Adam(params=params, lr=self.args.learning_rate)

            training_loss = self.step(data=self.train_data,
                                      optimizer=optimizer)
            with torch.no_grad():
                test_loss = self.step(data=self.val_data)

            if test_loss < min_loss:
                self.model.save_weights(str(epoch + 1))
                min_loss = test_loss

            writer.add_scalars('loss', {'train': training_loss, 'val': test_loss}, epoch + 1)

            print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, TestLoss: {:.4f}, TestPerplexity: {:5.4f}'
                  .format(epoch + 1, self.args.num_epochs, training_loss,
                          np.exp(training_loss), test_loss, np.exp(test_loss)))

            self.args.learning_rate *= 0.995

        if self.args.save_weight:
            self.model.save_weights()

    def step(self, data, optimizer=None):
        if optimizer is None:
            data_loader = DataLoader(data, batch_size=self.args.batch_size, shuffle=False)
            batch_size = self.args.batch_size
        else:
            data_loader = DataLoader(data, batch_size=self.args.batch_size, shuffle=True)
            batch_size = self.args.batch_size

        total_loss = 0
        step = 0
        num_of_step = int((data_loader.__len__()))

        for x, y in data_loader:
            transform = transforms.Compose([
                id_to_trainId_map_func
            ])
            y = torch.from_numpy(transform(y))

            outputs = self.model(x.permute(0, 3, 1, 2).float().to(self.device))

            if self.loss_balance:
                criterion = nn.CrossEntropyLoss(weight=self.weights.to(self.device))
            else:
                criterion = nn.CrossEntropyLoss()

            loss = criterion(outputs, y.long().to(self.device))
            total_loss += (loss.item() * self.args.batch_size)
            torch.cuda.empty_cache()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (step + 1) % self.args.batch_loss_print == 0:
                print('step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(step + 1, num_of_step, loss.item(),
                              np.exp(loss.item())))
            step += 1

        return total_loss / (num_of_step * batch_size)

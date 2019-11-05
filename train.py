import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from util import save_weights, load_weights


class Trainer:
    def __init__(self, args, device, encoder, decoder, training_data, val_data):
        self.args = args
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_input, self.train_output = training_data
        self.test_input, self.test_output = val_data

    def train(self):
        writer = SummaryWriter()

        if self.args.load_weight:
            load_weights(self.encoder, self.args.model_path + "encoder")
            load_weights(self.decoder, self.args.model_path + "decoder")

        for epoch in range(self.args.num_epochs):
            params = list(self.decoder.parameters())
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=params, lr=self.args.learning_rate)

            training_loss = self.step(criterion=criterion,
                                      data=(self.train_input, self.train_output),
                                      optimizer=optimizer)

            with torch.no_grad():
                test_loss = self.step(criterion=criterion,
                                      data=(self.test_input, self.test_output))


            writer.add_scalars('loss', {'train': training_loss,
                                        'val': test_loss}, epoch + 1)

            print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, TestLoss: {:.4f}, TestPerplexity: {:5.4f}'
                  .format(epoch + 1, self.args.num_epochs, training_loss,
                          np.exp(training_loss), test_loss, np.exp(test_loss)))

            self.args.learning_rate *= 0.995

        if self.args.save_weight:
            save_weights(self.encoder, self.args.model_path + "encoder")
            save_weights(self.decoder, self.args.model_path + "decoder")

    def step(self, criterion, data, optimizer=None):
        data_inputs, data_outputs = data
        numofstep = int(data_inputs.shape[0] / self.args.batch_size)
        total_Loss = 0
        for step in range(numofstep):

            start = self.args.batch_size * step

            data_input = torch.from_numpy(data_inputs[start: start + self.args.batch_size]).type(torch.FloatTensor).to(
                self.device)
            data_output = torch.from_numpy(data_outputs[start: start + self.args.batch_size]).long().to(self.device)

            feature = self.encoder(data_input.permute(0, 3, 1, 2))

            outputs = self.decoder(feature)

            loss = criterion(outputs, data_output)

            total_Loss += (loss.item() * self.args.batch_size)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (step + 1) % 100 == 0:
                    print('step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                          .format(step + 1, numofstep, loss.item(),
                                  np.exp(loss.item())))
        return total_Loss / (numofstep * self.args.batch_size)

import torch
from label import label2color
from PIL import Image
import numpy as np
import random


class Sample:
    def __init__(self, args, device, encoder, decoder, training_data, val_data):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.args = args
        self.train_input, self.train_output = training_data
        self.test_input, self.test_output = val_data

    def random_sample(self):
        train_mask = [random.randint(0, self.train_input.shape[0] - 1) for _ in range(self.args.numOfpredection)]
        test_mask = [random.randint(0, self.test_input.shape[0] - 1) for _ in range(self.args.numOfpredection)]

        self.train_input = self.train_input[train_mask]
        self.test_input = self.test_input[test_mask]

        self.train_output = self.train_output[train_mask]
        self.test_output = self.test_output[test_mask]

        with torch.no_grad():
            Train_out = self.predict(self.train_input)
            Test_out = self.predict(self.test_input)

        _, Train_out = torch.max(Train_out, 1)
        _, Test_out = torch.max(Test_out, 1)
        print(self.train_input.shape, self.train_output.shape , Train_out.cpu().numpy().shape)
        self.Save((self.train_input, self.train_output, Train_out.cpu().numpy()), "Sample/train/")
        self.Save((self.test_input, self.test_output, Test_out.cpu().numpy()), "Sample/val/")

    def predict(self, data_inputs):
        numofstep = int(data_inputs.shape[0] / 2)
        outputs = []
        for step in range(numofstep):
            start = 2 * step
            data_input = torch.from_numpy(data_inputs[start: start + 2]).type(torch.FloatTensor).to(
                self.device)
            feature = self.encoder(data_input.permute(0, 3, 1, 2))
            output = self.decoder(feature)
            outputs.append(output[0])
            outputs.append(output[1])
        outputs = torch.stack(outputs, 0)
        return outputs

    def Save(self, data, path):
        numberOFdaat = data[0].shape[0]
        for i in range(numberOFdaat):
            image = []
            for j in range(len(data)):
                if not j == 0:
                    img = label2color(data[j][i])
                else:
                    img = data[j][i]
                image.append(img)
            image = np.concatenate(image, 1)
            image = Image.fromarray(image.astype(np.uint8))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(path + str(i) + ".JPG")

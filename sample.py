import torch
from PIL import Image
import numpy as np
import random
import os
from label import traindId_to_color_map_func, id_to_trainId_map_func


class Sample:
    def __init__(self, args, device, model, training_data, val_data):
        self.model = model.to(device)
        self.device = device
        self.args = args
        # self.train_input, self.train_output = training_data
        # self.test_input, self.test_output = val_data

    def random_sample(self, train, val):
        train_mask = [random.randint(0, train.__len__() - 1) for _ in range(self.args.numOfpredection)]
        val_mask = [random.randint(0, val.__len__() - 1) for _ in range(self.args.numOfpredection)]

        train_list_input = []
        train_list_output = []
        for i in train_mask:
            x, y = train[i]
            train_list_input.append(np.array(x))
            train_list_output.append(np.array(id_to_trainId_map_func(y)))

        val_list_input = []
        val_list_output = []
        for i in val_mask:
            x, y = val[i]
            val_list_input.append(np.array(x))
            val_list_output.append(id_to_trainId_map_func(y))

        Train_out = self.predict(train_list_input)
        Val_out = self.predict(val_list_input)

        self.Save((train_list_input, train_list_output, Train_out.cpu().numpy()),
                  "Sample/" + self.model.name + "/train/")
        self.Save((val_list_input, val_list_output, Val_out.cpu().numpy()), "Sample/" + self.model.name + "/val/")

    def predict(self, data_inputs):
        with torch.no_grad():
            shape = (len(data_inputs), 1024, 2048)
            outputs = torch.zeros(shape)

            for i, x in enumerate(data_inputs):
                image = self.model(
                    torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(
                        self.device))
                _, image = torch.max(image, 1)

                outputs[i] = image

        return outputs

    def Save(self, data, path):
        os.makedirs(path)
        number_of_data = len(data[0])
        c = 0
        for i in range(number_of_data):
            c += 1
            print(c)
            inp, out, pre = data[0][i], data[1][i], data[2][i]
            out = traindId_to_color_map_func(out)
            pre = traindId_to_color_map_func(pre)

            out = np.stack(out, 0)
            out = np.transpose(out, (1, 2, 0))

            pre = np.stack(pre, 0)
            pre = np.transpose(pre, (1, 2, 0))

            image = np.concatenate((inp, out, pre), 1)
            image = Image.fromarray(image.astype(np.uint8))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(path + str(i) + ".JPG")

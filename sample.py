import torch
import random, os
import numpy as np
from PIL import Image
from label import traindId_to_color_map_func, id_to_trainId_map_func


class Sample:
    def __init__(self, args, device, model):
        self.model = model.to(device)
        self.device = device
        self.args = args

    def random_sample(self, train, val):
        train_mask = [random.randint(0, train.__len__() - 1) for _ in range(self.args.number_Of_predection)]
        val_mask = [random.randint(0, val.__len__() - 1) for _ in range(self.args.number_Of_predection)]

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

        train_out = self.predict(train_list_input)
        val_out = self.predict(val_list_input)

        self.save((train_list_input, train_list_output, train_out.cpu().numpy()),
                  "Sample/" + self.model.name + "/train/")
        self.save((val_list_input, val_list_output, val_out.cpu().numpy()), "Sample/" + self.model.name + "/val/")
        f = open("Sample/" + self.model.name + "/args.txt", "w+")
        f.writelines("epochs :" + str(self.args.num_epochs) + "\nLR :" + str(self.args.learning_rate))

        # print("hellllllllooooo")

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

    def save(self, data, path):
        os.makedirs(path)

        number_of_data = len(data[0])
        c = 0

        for i in range(number_of_data):
            c += 1
            print(c)
            inp, out, pre = data[0][i], data[1][i], data[2][i]

            out = np.transpose(np.stack(traindId_to_color_map_func(out), 0), (1, 2, 0))
            pre = np.transpose(np.stack(traindId_to_color_map_func(pre), 0), (1, 2, 0))

            image = np.concatenate((inp, out, pre), 1)
            image = Image.fromarray(image.astype(np.uint8))

            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(path + str(i) + ".JPG")

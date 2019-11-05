import os, sys
from PIL import Image
import numpy as np
import torch
from label import color2label

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


def load_data(array=True, val=False):
    if val:
        path = "/home/ziad/Documents/Semantic Segmentation/CityScapes/val/"
    else:
        path = "/home/ziad/Documents/Semantic Segmentation/CityScapes/train/"

    numOFimage = len([f for f in os.listdir(path + "input/") if os.path.isfile(os.path.join(path + "input/", f))])
    input_images = []
    output_images = []

    for i in range(1, numOFimage + 1):
        raw_image = Image.open(path + "input/" + str(i) + ".png")
        input_images.append(np.asarray(raw_image))
        if not array:
            output = Image.open(path + "output/" + str(i) + ".png")
            output_images.append(color2label(np.array(output)))
        if i % 100 == 0:
            print(i, "image loaded")

    if array:
        output_images = np.load(path + "output.npy")
    else:
        output_images = np.array(output_images)
        np.save(path + "output.npy", output_images)

    input_images = np.array(input_images)
    return input_images, output_images

def load_weights(model, path):
    model.load_state_dict(torch.load(os.path.join(path)))


def save_weights(model, path):
    torch.save(model.state_dict(), os.path.join(path))
import os
import cv2
import numpy as np
from PIL import Image


def load_images_from_folder(val=False):
    if val:
        root_folder = '/home/ziad/Documents/Semantic-Segmentation/cityScape/leftImg8bit_trainvaltest/leftImg8bit/val'
    else:
        root_folder = '/home/ziad/Documents/Semantic-Segmentation/cityScape/leftImg8bit_trainvaltest/leftImg8bit/train'

    inputs = []
    outputs = []
    count = 0
    folder_gen = os.walk(root_folder)
    folders = next(folder_gen)[1]
    for folder in folders:
        images_gen = os.walk(root_folder + "/" + folder)
        for img in next(images_gen)[2]:
            path = root_folder + "/" + folder + "/" + img
            gt_path = path.replace("leftImg8bit", "gtFine").replace(".png", "_color.png")

            inputs.append(cv2.imread(path))
            outputs.append(cv2.imread(gt_path))
            count += 1
            if count % 100 == 0:
                print(count)
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    print(inputs.shape)
    print(outputs.shape)

    return inputs, outputs


load_images_from_folder(True)

import os
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms

dir = "data/"


class Type:
    def __init__(self, images, labels, haircut_index):
        self.images = images
        self.labels = labels
        self.haircut_index = haircut_index

    def get_images(self):
        return self.images

    def get_labels(self):
        return self.labels

    def get_haircut_indexes(self):
        return self.haircut_index


def get_images_and_labels():
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    train_set = []
    test_set = []
    labels_train = pd.read_csv(dir + "train_labels.csv")
    labels_test = pd.read_csv(dir + "test_labels.csv")
    for i in os.listdir(dir + "train_segmented_black_white"):
        name = os.path.join(dir + "train_segmented_black_white", i)
        a = int(labels_train[(labels_train == i).any(axis=1)].type)
        with Image.open(name) as img:
            train_set.append([tfms(Image.open(name)).unsqueeze(0)[0], a])

    for i in os.listdir(dir + "test_segmented_black_white"):
        name = os.path.join(dir + "test_segmented_black_white", i)
        a = int(labels_test[(labels_test == i).any(axis=1)].type)
        with Image.open(name) as img:
                test_set.append([tfms(Image.open(name)).unsqueeze(0)[0], a])

    import random
    random.shuffle(train_set)
    random.shuffle(test_set)
    haircuts = pd.read_csv(dir + "haircuts.csv")
    return Type(train_set, labels_train, haircuts), Type(test_set, labels_test, haircuts)

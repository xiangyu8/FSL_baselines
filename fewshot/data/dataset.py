# This code is modified from
# https://github.com/facebookresearch/low-shot-shrink-hallucinate

import glob
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from numpy import random
from PIL import Image
import torchfile

def identity(x):
    return x


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img

class SetDataset:
    def __init__(
        self,
        data_file,
        batch_size,
        transform,
        args=None,
    ):
        with open(data_file, "r") as f:
            self.meta = json.load(f)

        self.args = args
        self.cl_list = np.unique(self.meta["image_labels"]).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []


        for x, y in zip(self.meta["image_names"], self.meta["image_labels"]):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # use main thread only or may receive multiple batches
            pin_memory=False,
        )
        for i, cl in enumerate(self.cl_list):
            sub_dataset = SubDataset(
                self.sub_meta[cl],
                cl,
                transform=transform,
                args=self.args,
            )
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)
            )

    def __getitem__(self, i):
#        print("inside SetDataset: ", self.sub_dataloader[i].shape)
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)


class SubDataset:
    def __init__(
        self,
        sub_meta,
        cl,
        transform=transforms.ToTensor(),
        target_transform=identity,
        args=None,
    ):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        cl_path = os.path.split(self.sub_meta[0])[0]
        self.img = dict(np.load(os.path.join(cl_path, "img.npz")))

        # Used if sampling from class
        self.args = args

    def __getitem__(self, i):
        # To get image data
        image_path = self.sub_meta[i]
        img = self.img[image_path]
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target# return tensors, img, and the label

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[: self.n_way]

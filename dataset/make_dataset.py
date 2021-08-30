# -*- coding: utf-8 -*-
import h5py, os
import torch, cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset.pancreas import *


class make_data_3d(Dataset):
    def __init__(self, imgs, plabs, masks, labs):
        self.img = [img.cpu().squeeze().numpy() for img in imgs]
        self.plab = [np.squeeze(lab.cpu().numpy()) for lab in plabs]
        self.mask = [np.squeeze(mask.cpu().numpy()) for mask in masks]
        self.lab = [np.squeeze(lab.cpu().numpy()) for lab in labs]
        self.num = len(self.img)
        self.tr_transform = Compose([
            # RandomRotFlip(),
            CenterCrop((96, 96, 96)),
            # RandomNoise(),
            ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.img[idx], self.plab[idx], self.mask[idx], self.lab[idx]
        samples = self.tr_transform(samples)
        imgs, plabs, masks, labs = samples
        return imgs, plabs.long(), masks.float(), labs.long()

    def __len__(self):
        return self.num

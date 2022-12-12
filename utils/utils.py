#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:01:18 2022

@author: ishibashiryuto
"""

import torch
import cv2
import glob
import os
from torchvision import transforms


transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])

def Make_dataset(path, batch_size=16):
    data = []
    labels = []
    files = glob.glob(os.path.join(path,'*','*.*'))
    classes = os.listdir(path)
    for file in files:
        _cls = file.split('/')[-2]
        img = cv2.imread(file)
        img = transform(img)
        data.append(img.unsqueeze(0))
        labels.append(classes.index(_cls))
    data = torch.cat(data, dim=0)
    labels = torch.tensor(labels, dtype=torch.int64)
    dataset = torch.utils.data.TensorDataset(data,labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    return loader, classes


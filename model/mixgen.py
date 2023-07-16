"""
MixGen: A New Multi-Modal Data Augmentation
https://arxiv.org/abs/2206.08358
Apache-2.0 License, Copyright 2022 Amazon
"""
import random
import numpy as np
import torch
from torchvision import transforms


def mixgen(image, text, num, lam=0.5):
    # default MixGen
    for i in range(num):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[i+num,:]
        # text concat
        text[i] = text[i] + " " + text[i+num]
    return image, text

def concatgen(image, text, num, lam=0.5):
    for i in range(num):
        # image mixup
        img1 = transforms.functional.resize(image[i], (224, 112))
        img2 = transforms.functional.resize(image[i+num], (224, 112))
        image[i] = torch.cat((img1, img2), dim=2)
        image[i] = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image[i])
        # text concat
        text[i] = text[i] + " " + text[i+num]
    return image, text


def mixgen_batch(image, text, num, lam=0.5):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)
    for i in range(batch_size):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[index[i],:]
        # text concat
        text[i] = text[i] + " " + text[index[i]]
    return image, text

import json
import os
import random

import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms

from misc.caption_dataset import ps_train_dataset, ps_eval_dataset
from misc.utils import is_using_distributed


def get_self_supervised_augmentation(img_size):
    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    aug = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.), antialias=True),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return aug


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class cuhkpedes_eval(torch.utils.data.Dataset):
    def __init__(self, ann_file, transform, image_root):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.pid2txt, self.pid2img = {}, {}
        self.txt_ids, self.img_ids = [], []

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            if ann['image_id'] not in self.pid2txt.keys():
                self.pid2txt[ann['image_id']] = []
                self.pid2img[ann['image_id']] = []
            self.pid2img[ann['image_id']].append(img_id)
            self.img_ids.append(ann['image_id'])
            for i, caption in enumerate(ann['caption']):
                self.text.append(caption)
                self.pid2txt[ann['image_id']].append(txt_id)
                self.txt_ids.append(ann['image_id'])
                txt_id += 1

        for tid in range(len(self.text)):
            self.txt2img[tid] = self.pid2img[self.txt_ids[tid]]
        for iid in range(len(self.image)):
            self.img2txt[iid] = self.pid2txt[self.img_ids[iid]]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path)
        image = self.transform(image)

        return image, index


def build_pedes_data(config):
    size = config.experiment.input_resolution
    if isinstance(size, int):
        size = (size, size)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        normalize
    ])

    rand_from = [
        transforms.ColorJitter(.1, .1, .1, 0),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(size, (0.9, 1.0), antialias=True),
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.10, 0.20)),
    ]
    aug = Choose(rand_from, size)
    aug_ss = get_self_supervised_augmentation(size)

    train_dataset = ps_train_dataset(config.anno_dir, config.image_dir, aug, aug_ss, split='train', max_words=77)
    test_dataset = ps_eval_dataset(config.anno_dir, config.image_dir, val_transform, split='test', max_words=77)

    if is_using_distributed():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    test_sampler = None

    config_data = config.data
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config_data.batch_size,
        shuffle=train_sampler is None,
        num_workers=config_data.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )

    return {
        'train_loader': train_loader,
        'train_sampler': train_sampler,
        'test_loader': test_loader,
    }


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class Choose:
    def __init__(self, rand_from, size):
        self.choose_from = rand_from
        self.size = size

    def __call__(self, image):
        aug_choice = np.random.choice(self.choose_from, 2)
        return transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            *aug_choice,
            normalize
        ])(image)

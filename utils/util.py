import os
import torch
import json
import pdb
import albumentations
import numpy as np
import torch.nn as nn
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class EditingJsonDataset(Dataset):
    def __init__(self, args, repeats=1):
        self.image_dir = args.image_dir_path
        self.transform = build_transform(args)
        with open(args.json_file, 'r') as f:
            self.image_prompt = json.load(f)
            self.image_files = list(self.image_prompt.keys())*repeats
        f.close()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        
        original_prompt, editing_prompt = self.image_prompt[self.image_files[idx]][0], self.image_prompt[self.image_files[idx]][1]
        if self.transform:
            image = self.transform(image)

        return image, original_prompt, editing_prompt
    
class EditingSingleImageDataset(Dataset):
    def __init__(self, args, repeats=1):
        self.transform = build_transform(args)
        self.image_files = [args.image_file_path] * repeats
        self.image_prompt = [args.image_caption, args.editing_prompt]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)

        original_prompt, editing_prompt = self.image_prompt[0], self.image_prompt[1]
        if self.transform:
            image = self.transform(image)

        return image, original_prompt, editing_prompt

def build_dataset(args, data_path):
    transform = build_transform(is_train, args)
    root = data_path
    dataset = datasets.ImageFolder(root, transform=transform)

    return datase

def build_transform(args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    t = []
    if args.image_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.image_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC, antialias=True)  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.image_size))

    t.append(transforms.ToTensor())

    return transforms.Compose(t)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    new_sample = images["new_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(new_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()

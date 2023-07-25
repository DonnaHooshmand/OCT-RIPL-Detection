"""
Data Generator
"""
import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

def parse_image(img_path, image_h, image_w, noise):
    image_rgb = (cv2.imread(img_path, 1)/255).astype(np.float32)

    h, w = image_rgb.shape
    if (h == image_h) and (w == image_w):
        pass
    else:
        image_rgb = cv2.resize(image_rgb, (image_w, image_h))
    if noise==True:
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        noisy_image = add_gaussian_noise(gray_image, mean=0, std_dev=np.var(gray_image)*2)
        # noisy_image = add_gaussian_noise(gray_image, mean=0, std_dev=3)
        std_image = noisy_image
        return std_image
    else:
        return image_rgb

def parse_mask(mask_path, image_h, image_w):
    mask = cv2.imread(mask_path, -1)
    h, w = mask.shape
    if (h == image_h) and (w == image_w):
        pass
    else:
        mask = cv2.resize(mask, (image_w, image_h))
    mask = np.expand_dims(mask, -1)
    mask = mask/255.0
    return mask

def add_gaussian_noise(image, mean, std_dev):
    # Generate random Gaussian noise
    noise = np.random.normal(mean, std_dev, image.shape)

    # Add noise to the image
    # noisy_image = cv2.add(image, noise)
    noisy_image = image+noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image



class DataGen(Dataset):
    def __init__(self, image_h, image_w, images_path, masks_path, noise=False):
        self.image_h = image_h
        self.image_w = image_w
        self.images_path = images_path
        self.masks_path = masks_path
        # self.batch_size = batch_size
        self.noise = noise

    def __getitem__(self, index):
        
        image = parse_image(self.images_path[index], self.image_h, self.image_w, self.noise)
        # print(f'index {index} image shape {image.shape}')
        mask = parse_mask(self.masks_path[index], self.image_h, self.image_w)
        # print(f'index {index} mask shape {mask.shape}')

        return image, mask

    def __len__(self):
        return len(self.images_path)

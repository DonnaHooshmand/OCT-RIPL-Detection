
import os
import numpy as np
import cv2
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix

from data_generator import DataGen
# from unet import Unet
# from resunet import ResUnet
from resunetPlusPlus_pytorch import build_resunetplusplus
# from metrics import dice_coef, dice_loss

from tensorflow.keras.optimizers import Adam, Nadam, SGD

if __name__ == "__main__":
    ## Path
    file_path = "files/"
    model_path = "files/resunetplusplus.h5"

    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass

    train_path = "new_data/kvasir_segmentation_dataset/train/"
    valid_path = "new_data/kvasir_segmentation_dataset/valid/"

    ## Training
    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    # train_image_paths = train_image_paths[:2000]
    # train_mask_paths = train_mask_paths[:2000]

    ## Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()

    ## Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 200

    train_steps = len(train_image_paths)//batch_size
    valid_steps = len(valid_image_paths)//batch_size

    ## Generator
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)

    ## Turn the data into a torch.utils.data thing
    train_loader = torch.utils.data.DataLoader(train_gen, batch_size=1000, shuffle=True)
    
    ## ResUnet++
    model = build_resunetplusplus()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

breakpoint()
# The training loop
for epoch in range(1):
    total_correct = 0
    total_loss = 0
    for batch in train_loader:
        images, labels = batch
        
        optimizer.zero_grad()
        preds = model(images)
        
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step( )
        
        total_loss += loss.item()
        total_correct += preds.argmax(dim=1).eq(labels).sum().item()

    print('epoch:', epoch, "total_correct:", total_correct, "loss:", total_loss)




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

import os
import numpy as np
import cv2
from glob import glob

from pytorch_datagen import DataGen
from resunetPlusPlus_pytorch_copy import build_resunetplusplus

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
    print("train steps: ", train_steps)
    valid_steps = len(valid_image_paths)//batch_size
    print("valid steps: ", valid_steps)
    
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths)
    
    ## Turn the data into a torch.utils.data thing
    train_loader = torch.utils.data.DataLoader(train_gen, batch_size=8)
    valid_loader = torch.utils.data.DataLoader(valid_gen, batch_size=8)
    
    ## ResUnet++
    model = build_resunetplusplus()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
    # The training loop
    for epoch in range(train_steps):
        total_correct = 0
        total_loss = 0
        for batch in train_loader:
            images, labels = batch
            images = images.to(torch.float)
            labels = labels.to(torch.float)
            

            optimizer.zero_grad()
            images = images.unsqueeze(1)
            labels = torch.permute(labels, (0,3,1,2))

            preds = model(images)
            
            loss = F.cross_entropy(preds, labels)
            loss.backward()
            optimizer.step( )

            total_loss += loss.item()
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()

        print('epoch:', epoch, "total_correct:", total_correct, "loss:", total_loss)




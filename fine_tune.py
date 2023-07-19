import pickle
from resunetPlusPlus_pytorch_1channel import build_resunetplusplus

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

from pytorch_datagen_finetune import DataGen

import os
import shutil
import random


def split_dataset(image_dir, mask_dir, train_dir, train_mdir, 
                  validation_dir, validation_mdir, test_dir, 
                  test_mdir, split_ratios=(0.8, 0.1, 0.1)):
    # Create target directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_mdir, exist_ok=True)
    os.makedirs(validation_mdir, exist_ok=True)
    os.makedirs(test_mdir, exist_ok=True)

    # Get a list of all files in the source directory
    file_list = os.listdir(image_dir)

    # Shuffle the file list to randomize the order
    random.shuffle(file_list)

    # Calculate the number of files for each split
    num_files = len(file_list)
    num_train = int(split_ratios[0] * num_files)
    num_validation = int(split_ratios[1] * num_files)

    # Split the dataset
    train_files = file_list[:num_train]
    validation_files = file_list[num_train:num_train + num_validation]
    test_files = file_list[num_train + num_validation:]

    # Move images and masks to their respective split folders
    for files, target_dir, target_mdir in [(train_files, train_dir, train_mdir), 
                                           (validation_files, validation_dir, validation_mdir), 
                                           (test_files, test_dir, test_mdir)]:
        for file in files:
            image_path = os.path.join(image_dir, file)
            mask_file_name = file.split('.')[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_file_name)  # Replace 'path_to_masks' with the actual path to the masks
            shutil.copy(image_path, target_dir)
            shutil.copy(mask_path, target_mdir)  # Adjust this line if the masks are in a different folder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU available: ", torch.cuda.is_available())

model_path = r'ColonoscopyTrained_resUnetPlusPlus.pkl'    
model = build_resunetplusplus()
model.load_state_dict(torch.load(model_path))
model.to(device)


for name, parameter in model.named_parameters():
    if 'output' in name:
        print(f"parameter '{name}' will not be freezed")
        parameter.requires_grad = True
    else:
        parameter.requires_grad = False



# start training
image_path = r'data/RIPL_data/RIPL_all'
mask_path = r'data/RIPL_data/masks'

train_dir =  r'data/RIPL_data/train_img'
train_mdir = r'data/RIPL_data/train_mask' 
validation_dir = r'data/RIPL_data/valid_img'
validation_mdir = r'data/RIPL_data/valid_mask'
test_dir = r'data/RIPL_data/test_img'
test_mdir = r'data/RIPL_data/test_mask'

split_dataset(image_path, mask_path, train_dir, train_mdir, validation_dir, validation_mdir, test_dir, test_mdir)

 ## Training
train_image_paths = glob(os.path.join(train_dir, "*"))
train_mask_paths = glob(os.path.join(train_mdir, "*"))
train_image_paths.sort()
train_mask_paths.sort()

## Validation
valid_image_paths = glob(os.path.join(validation_dir, "*"))
valid_mask_paths = glob(os.path.join(validation_mdir, "*"))
valid_image_paths.sort()
valid_mask_paths.sort()

## Parameters
image_h = 496
image_w = 768
batch_size = 8
lr = 1e-4
epochs = 200

train_steps = len(train_image_paths)//batch_size
print("train steps: ", train_steps)
valid_steps = len(valid_image_paths)//batch_size
print("valid steps: ", valid_steps)

train_gen = DataGen(image_h, image_w, train_image_paths, train_mask_paths)
valid_gen = DataGen(image_h, image_w, valid_image_paths, valid_mask_paths)

## Turn the data into a torch.utils.data thing
train_loader = torch.utils.data.DataLoader(train_gen, batch_size=8)
valid_loader = torch.utils.data.DataLoader(valid_gen, batch_size=8)
# for image, label in train_loader:
#     displayTensor(image[0], "testing_inputs.png")
#     raise ValueError

## ResUnet++


model = build_resunetplusplus()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_tracker = {}

# The training loop
for epoch in range(150):
    total_correct = 0
    # t_accuracy = 0
    total_loss = 0
    n = 0
    for t, batch in enumerate(train_loader):
        if n != 9:
            n+=1
            images, labels = batch
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()
            images = images.unsqueeze(1).to(device)
            labels = labels.permute(0, 3, 1, 2).to(device)
            # images = images.permute(0, 3, 1, 2).to(device)
            print('the dimensions of labels 1 is: ', labels.shape)
            print('the dimensions of image 1 is: ', images.shape)

            # print('the dimensions of the input image is: ', images.shape)
            preds = model(images)
            print('the dimensions of preds is: ', preds.shape)
            print('the dimensions of labels is: ', labels.shape)
            loss = F.mse_loss(preds, labels).to(device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # t_accuracy += dice_coeff(preds, labels), preds.size(0)
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()

            print("finished batch ", n, " for epoch ", epoch)
        else:
            n+=1
        
    # Validation phase
    model.eval()
    valid_loss = 0
    valid_correct = 0
    # v_accuracy = 0
    with torch.no_grad():
        for v, batch in enumerate(valid_loader):
            images, labels = batch
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            images = images.unsqueeze(1).to(device)
            labels = labels.permute(0, 3, 1, 2).to(device)
            # images = images.permute(0, 3, 1, 2).to(device)

            preds = model(images)
            
            loss = F.mse_loss(preds, labels).to(device)
            # loss = loss_type(preds, labels).to(device)
            print('validation loss: ', loss)

            valid_loss += loss.item()
            # v_accuracy += dice_coeff(preds, labels), preds.size(0)
            valid_correct += preds.argmax(dim=1).eq(labels).sum().item()

    # Calculate average losses and accuracies
    train_loss = total_loss / (t+1)
    train_accuracy = total_correct / (t+1)
    valid_loss = valid_loss / (v+1)
    valid_accuracy = valid_correct / (v+1)

    if epoch % 10 == 0:
        epoch_tracker[epoch] = ['training loss: ', train_loss, 'training accuracy: ', train_accuracy, 'validation loss: ', valid_loss, 'validation accuracy: ', valid_accuracy]
        
    # Print or store the results
    print('-------------------Epoch:', epoch)
    print('Training - Loss:', train_loss, 'Accuracy:', train_accuracy)
    print('Validation - Loss:', valid_loss, 'Accuracy:', valid_accuracy)

    # Switch back to training mode
    model.train()
print(epoch_tracker)
torch.save(model.state_dict(), 'finetuned_resUnetPlusPlus_gb.pkl')




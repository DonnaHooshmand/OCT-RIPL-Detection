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

from pytorch_datagen_resize import DataGen
from resunetPlusPlus_pytorch_1channel import build_resunetplusplus
import pandas as pd



def displayTensor(input_img: torch.tensor, file_name) -> None:
    """
    Display a tensor as an image using matplotlib.
    """
    input_img_cpu = input_img.detach().cpu().numpy()
    input_img_cpu = np.squeeze(input_img_cpu)
    plt.imsave(file_name,input_img_cpu, cmap='gray')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU available ", torch.cuda.is_available())

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

    ## Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
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
    
    train_gen = DataGen(image_h, image_w, train_image_paths, train_mask_paths, noise=True)
    valid_gen = DataGen(image_h, image_w, valid_image_paths, valid_mask_paths, noise=True)
    
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
    


    train_losses = []
    val_losses = []

    # The training loop
    for epoch in range(150):
        
        total_loss = 0
        n = 0
        for t, batch in enumerate(train_loader):
            n+=1
            images, labels = batch
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()
            # images = images.unsqueeze(1).to(device)
            labels = labels.permute(0, 3, 1, 2).to(device)
            images = images.permute(0, 3, 1, 2).to(device)

            preds = model(images)
            
            loss = F.mse_loss(preds, labels).to(device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print("finished batch ", n, " for epoch ", epoch)
        
        # Validation phase
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for v, batch in enumerate(valid_loader):
                images, labels = batch
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                # images = images.unsqueeze(1).to(device)
                labels = labels.permute(0, 3, 1, 2).to(device)
                images = images.permute(0, 3, 1, 2).to(device)

                preds = model(images)
                
                loss = F.mse_loss(preds, labels).to(device)
                # loss = loss_type(preds, labels).to(device)
                
                valid_loss += loss.item()

        # Calculate average losses and accuracies
        train_loss = total_loss / (t+1)

        valid_loss = valid_loss / (v+1)
  
        val_losses.append(valid_loss)
        train_losses.append(train_loss)
           
        # Print or store the results
        print('-------------------Epoch:', epoch)
        print('Training - Loss:', train_loss)
        print('Validation - Loss:', valid_loss)

        # Switch back to training mode
        model.train()

    loss_dict = {'train_loss': train_losses, 'val_loss': val_losses}
    df = pd.DataFrame(loss_dict)
    df.to_csv('resize_colonoscopy_loss_softmax.csv')

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size

    # Plotting training loss
    plt.plot(epochs, train_losses, label='Training Loss', marker='', linestyle='-', color='blue')

    # Plotting validation loss
    plt.plot(epochs, val_losses, label='Validation Loss', marker='', linestyle='-', color='orange')

    # Add labels, title, and legend
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs for model trained on resized noisy colonoscopy images')
    plt.legend()


    plt.savefig('resized_colonoscopy_loss_plot.png')

    torch.save(model.state_dict(), 'resize_colonoscopy_trained_resUnetPlusPlus.pkl')


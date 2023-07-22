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
from resunetPlusPlus_pytorch_1channel_nosoftmax import build_resunetplusplus


def displayTensor(input_img: torch.tensor, file_name) -> None:
    """
    Display a tensor as an image using matplotlib.
    """
    input_img_cpu = input_img.detach().cpu().numpy()
    input_img_cpu = np.squeeze(input_img_cpu)
    plt.imsave(file_name,input_img_cpu, cmap='gray')


if __name__ == "__main__":
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


    validation_dir = r'data/RIPL_data/valid_img'
    validation_mdir = r'data/RIPL_data/valid_mask'

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

    valid_steps = len(valid_image_paths)//batch_size
    print("valid steps: ", valid_steps)

    valid_gen = DataGen(image_h, image_w, valid_image_paths, valid_mask_paths)

    ## Turn the data into a torch.utils.data thing
    valid_loader = torch.utils.data.DataLoader(valid_gen, batch_size=8)

    val_losses = []
    # The training loop
    for epoch in range(150):
        
        # Validation phase
        model.eval()
        valid_loss = 0
        
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
                print('preds: ', preds)
                
                loss = F.mse_loss(preds, labels).to(device)
                # loss = loss_type(preds, labels).to(device)
                

                valid_loss += loss.item()
                print('v value in loop: ', v)
            print('v value after loop: ', v)
        print('v value out of no grad: ', v)                

        # Calculate average losses and accuracies
        valid_loss = valid_loss / (v+1)
        
        val_losses.append(valid_loss)
           
        # Print or store the results
        print('-------------------Epoch:', epoch)
       
        print('Validation - Loss:', valid_loss)

        # Switch back to training mode
        model.train()

    print('unique values in val_losses: ', set(val_losses))
    
    torch.save(model.state_dict(), 'OGtrained_resUnetPlusPlus.pkl')


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# from sklearn.metrics import confusion_matrix
# from resources.plotcm import plot_confusion_matrix

import os
import numpy as np
import cv2
from glob import glob

import shutil
import random

from pytorch_datagen_threshold import DataGen
from resunetPlusPlus_pytorch_1channel import build_resunetplusplus


def displayTensor(input_img: torch.tensor, file_name) -> None:
    """
    Display a tensor as an image using matplotlib.
    """
    input_img_cpu = input_img.detach().cpu().numpy()
    input_img_cpu = np.squeeze(input_img_cpu)
    plt.imsave(file_name,input_img_cpu, cmap='gray')

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




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU available: ", torch.cuda.is_available())

    model_path = r'colonoscopy_noisy\threshold_trained_resUnetPlusPlus.pkl'    
    model = build_resunetplusplus()
    model.load_state_dict(torch.load(model_path))
    model.to(device)



    valid_path = "new_data/kvasir_segmentation_dataset/valid/"



    ## Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()
    
    ## Parameters
    # image_h = 496
    # image_w = 768
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 200
 
    valid_steps = len(valid_image_paths)//batch_size
    print("valid steps: ", valid_steps)
    

    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, noise=True)
    
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
                # print("v: ", v, " batch: ", batch)
                images, labels = batch
                # plt.hist(np.ndarray.flatten(np.array(labels)))
                # plt.ylim(0, 10)
                # plt.show()
                
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                
                images = images.unsqueeze(1).to(device)
                labels = labels.permute(0, 3, 1, 2).to(device)
                # images = images.permute(0, 3, 1, 2).to(device)
                
                preds = model(images)
                # print('preds: ', preds)
                print('unique values in preds: ', torch.unique(preds))
                print('unique values in labels: ', torch.unique(labels))
                
                
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
    
    # torch.save(model.state_dict(), 'OGtrained_resUnetPlusPlus.pkl')


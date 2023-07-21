

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
# from resources.plotcm import plot_confusion_matrix

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

from pytorch_datagen import DataGen
from pytorch_datagen import *
from resunetPlusPlus_pytorch_copy import build_resunetplusplus



def displayTensor(input_img: torch.tensor, file_name) -> None:
    """
    Display a tensor as an image using matplotlib.
    """
    input_img_cpu = input_img.detach().cpu().numpy()
    input_img_cpu = np.squeeze(input_img_cpu)
    plt.imsave(file_name,input_img_cpu, cmap='gray')


def display_all(img, truth,  pred, file_name):
    img_cpu = img.detach().cpu().numpy()
    img_cpu = np.squeeze(img_cpu)
    truth_cpu = truth.detach().cpu().numpy()
    truth_cpu = np.squeeze(truth_cpu)
    pred_cpu = pred.detach().cpu().numpy()
    pred_cpu = np.squeeze(pred_cpu)

    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,10))

    ax[0].imshow(img_cpu)
    ax[0].set_title("Original Image")
    ax[1].imshow(truth_cpu)
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_cpu)
    ax[2].set_title("Prediction")
    figure.tight_layout()
    figure.savefig(file_name)

def overlayDisplay(input_img: torch.tensor, mask1: torch.tensor, mask2: torch.tensor) -> None:
    color1 = np.array([0,0,1]).astype(np.float32)
    color2 = np.array([1,0,0]).astype(np.float32)
    input_img_cpu = np.squeeze(input_img.clone().detach().cpu().numpy()).astype(np.float32)
    gray_img = np.stack((input_img_cpu, input_img_cpu, input_img_cpu), axis=2)
    mask1_cpu = np.squeeze(mask1.clone().detach().cpu().numpy()).astype(np.float32)
    mask2_cpu = np.squeeze(mask2.clone().detach().cpu().numpy()).astype(np.float32)
    mask1_cpu = np.where(mask1_cpu[...,None], color1, gray_img)
    mask2_cpu = np.where(mask2_cpu[...,None], color2, gray_img)
    out = cv2.addWeighted(mask1_cpu, 0.5, mask2_cpu, 0.5,0)
    out = cv2.addWeighted(gray_img, 0.2, out, 0.8,0)
    plt.imsave("all_display.png", out)

if __name__ == "__main__":
    model_path = "OGtrained_resUnetPlusPlus.pkl"
    save_path = "result/OG"
    test_path = "new_data/kvasir_segmentation_dataset/test/"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU available: ", torch.cuda.is_available())

    image_size = 256
    batch_size = 1

    test_image_paths = glob(os.path.join(test_path, "images", "*"))
    test_mask_paths = glob(os.path.join(test_path, "masks", "*"))
    test_image_paths.sort()
    test_mask_paths.sort()

    ## Create result folder
    try:
        os.mkdir(save_path)
    except:
        pass

    ## Model
    model = build_resunetplusplus()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
      
    test_steps = len(test_image_paths)//batch_size
    print("test steps: ", test_steps)
    test_gen = DataGen(image_size, test_image_paths, test_mask_paths)
    test_loader = torch.utils.data.DataLoader(test_gen)
    
    #switch off autograd
    with torch.no_grad():
        #det the model in evaluation mode
        model.eval()
        i=0
        for batch in test_loader:
            images, masks = batch
            
            images = images.to(device, dtype=torch.float)
            
            masks = masks.to(device, dtype=torch.float)
            
            # images = images.unsqueeze(1).to(device)
            masks = masks.permute(0, 3, 1, 2).to(device)
            images = images.permute(0, 3, 1, 2).to(device)
            
            predict_mask = model(images).to(device)
            # print("**", images.shape, images.dtype, masks.shape, masks.dtype, predict_mask.shape, predict_mask.dtype)
            
            loss = F.mse_loss(predict_mask, masks).to(device)
            
            
            predict_mask = (predict_mask > 0.5) * 255.0
            
            # displayTensor(images, "result/input_image.png")
            # displayTensor(predict_mask, "prediction.png")
            # displayTensor(masks, "truth.png")
            display_all(images, masks,  predict_mask, f"result/example{i}.png")
            overlayDisplay(images, masks, predict_mask)
            i+=1
            if i == 5:
                break


    print("done")


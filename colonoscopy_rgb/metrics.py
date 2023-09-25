import torch
import torch.nn.functional as F
import numpy as np


import torch.nn as nn
import torch.optim as optim


import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import os
import cv2
from glob import glob

import shutil
import random

from pytorch_datagen import DataGen
from resunetPlusPlus_pytorch_3channels import build_resunetplusplus

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# mIoU


def calculate_iou(predicted_masks, ground_truth_masks, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for a segmentation task.

    Args:
        predicted_masks (np.ndarray): Predicted binary masks (0 or 1).
        ground_truth_masks (np.ndarray): Ground truth binary masks (0 or 1).
        threshold (float, optional): Threshold for predicted masks. Default is 0.5.

    Returns:
        float: Mean IoU score.
    """
    predicted_masks = (predicted_masks > threshold).astype(np.uint8)
    ground_truth_masks = (ground_truth_masks > 0.5).astype(np.uint8)
    
    intersection = np.logical_and(predicted_masks, ground_truth_masks).sum()
    union = np.logical_or(predicted_masks, ground_truth_masks).sum()

    iou = intersection / (union + 1e-6)
    
    return iou

def calculate_miou(predicted_masks, ground_truth_masks, threshold=0.5):
    """
    Calculate Mean Intersection over Union (mIoU) for a segmentation task.

    Args:
        predicted_masks (list of np.ndarray): List of predicted binary masks.
        ground_truth_masks (list of np.ndarray): List of ground truth binary masks.
        threshold (float, optional): Threshold for predicted masks. Default is 0.5.

    Returns:
        float: Mean IoU score.
    """
    num_samples = len(predicted_masks)
    total_iou = 0.0

    for i in range(num_samples):
        iou = calculate_iou(predicted_masks[i], ground_truth_masks[i], threshold)
        total_iou += iou

    mean_iou = total_iou / num_samples

    return mean_iou



# Recall
# Precision

import numpy as np

def calculate_precision_recall(predicted_masks, ground_truth_masks, threshold=0.5):
    """
    Calculate Precision and Recall for a segmentation task.

    Args:
        predicted_masks (np.ndarray): Predicted binary masks (0 or 1).
        ground_truth_masks (np.ndarray): Ground truth binary masks (0 or 1).
        threshold (float, optional): Threshold for predicted masks. Default is 0.5.

    Returns:
        float: Precision
        float: Recall
    """
    predicted_masks = (predicted_masks > threshold).astype(np.uint8)
    ground_truth_masks = (ground_truth_masks > 0.5).astype(np.uint8)

    true_positives = np.logical_and(predicted_masks, ground_truth_masks).sum()
    false_positives = np.logical_and(predicted_masks, 1 - ground_truth_masks).sum()
    false_negatives = np.logical_and(1 - predicted_masks, ground_truth_masks).sum()

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    return precision, recall



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU available: ", torch.cuda.is_available())

    model_path = r'colonoscopy_noisy\threshold_trained_resUnetPlusPlus.pkl'    
    model = build_resunetplusplus()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    test_path = "new_data/kvasir_segmentation_dataset/test/"

    ## Testing
    test_image_paths = glob(os.path.join(test_path, "images", "*"))
    test_mask_paths = glob(os.path.join(test_path, "masks", "*"))
    test_image_paths.sort()
    test_mask_paths.sort()

    
    ## Parameters
    image_size = 256
    batch_size = 1
    lr = 1e-4
    epochs = 200
    
    test_steps = len(test_image_paths)//batch_size
    print("test steps: ", test_steps)
    
    test_gen = DataGen(image_size, test_image_paths, test_mask_paths)
    
    ## Turn the data into a torch.utils.data thing
    test_loader = torch.utils.data.DataLoader(test_gen, batch_size=batch_size)

    predicted_masks = []
    true_masks = []
    dice_losses = []
    reacalls = []
    precisions = []

    for v, batch in enumerate(test_loader):
        # print("v: ", v, " batch: ", batch)
        images, labels = batch
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        # images = images.unsqueeze(1).to(device)
        labels = labels.permute(0, 3, 1, 2).to(device)
        images = images.permute(0, 3, 1, 2).to(device)

        preds = model(images)
     
        predicted_masks.append(np.array(preds))
        true_masks.append(np.array(labels))

        dice_loss = dice_loss(preds, labels).to(device)
        dice_losses.append(dice_loss)

        precision, recall = calculate_precision_recall(np.array(preds), np.array(labels), threshold=0.5)
        recalls.append(recall)
        precisions.append(precision)
        
    miou = calculate_miou(predicted_masks, true_masks, threshold=0.5)
    average_dice_loss = sum(dice_losses) / len(dice_losses)
    average_recall = sum(recalls) / len(recalls)
    average_precision = sum(precisions) / len(precisions)
    print(f'miou: {miou}, average_dice_loss: {average_dice_loss}, average_recall: {average_recall}, average_precision: {average_precision}')

        
                      





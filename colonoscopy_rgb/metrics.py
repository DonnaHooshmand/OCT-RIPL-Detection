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

def calculate_iou(pred_masks, true_masks, num_classes):
    iou = []
    for class_idx in range(num_classes):
        intersection = np.sum((pred_masks == class_idx) & (true_masks == class_idx))
        union = np.sum((pred_masks == class_idx) | (true_masks == class_idx))
        iou_class = intersection / (union + 1e-10)  # Adding a small epsilon to avoid division by zero
        iou.append(iou_class)
    return iou

def calculate_miou(pred_masks, true_masks, num_classes):
    iou = calculate_iou(pred_masks, true_masks, num_classes)
    return sum(iou) / len(iou)

num_classes = 4
pred_masks = np.array([[0, 1, 1], [2, 2, 3]])
true_masks = np.array([[0, 1, 1], [2, 3, 3]])
miou = calculate_miou(pred_masks, true_masks, num_classes)
print(f"mIoU: {miou}")


# Recall
# Precision

def calculate_precision_recall(pred_masks, true_masks, class_idx):
    TP = np.sum((pred_masks == class_idx) & (true_masks == class_idx))
    FP = np.sum((pred_masks == class_idx) & (true_masks != class_idx))
    FN = np.sum((pred_masks != class_idx) & (true_masks == class_idx))

    precision = TP / (TP + FP + 1e-10)  # Adding a small epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)     # Adding a small epsilon to avoid division by zero

    return precision, recall

# Example usage:
class_idx = 1
precision, recall = calculate_precision_recall(pred_masks, true_masks, class_idx)
print(f"Class {class_idx}: Precision = {precision}, Recall = {recall}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU available: ", torch.cuda.is_available())

    model_path = r'colonoscopy_noisy\threshold_trained_resUnetPlusPlus.pkl'    
    model = build_resunetplusplus()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

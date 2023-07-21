from pytorch_datagen_finetune import parse_mask, parse_image
import os
import numpy as np
import cv2
from glob import glob

mask_path = r"C:\Users\collaborations\Desktop\Donna's_Folder\OCT-RIPL-Detection\data\RIPL_data\masks\00119ODYRsm.png"
image_path = r"C:\Users\collaborations\Desktop\Donna's_Folder\OCT-RIPL-Detection\data\RIPL_data\RIPL_all\00117ODYRs.jpg"

parse_image(image_path, 496, 768)


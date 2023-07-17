from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

path = r'/Users/lauramachlab/Library/CloudStorage/OneDrive-Personal/Documents/_northwestern/_MSAI/c3 lab/resunet_training/OCT-RIPL-Detection/data/RIPL_data/masks/'

# annotation = Image.open(path)
# annotation_array = np.array(annotation)
# unique_values = np.unique(annotation_array)
# print(type(unique_values))
# for value in unique_values:
#     if value != 0 and value != 255:
#         print(value)


mask_list = os.listdir(path)

# Loop through the segmentations
for i, mask_name in enumerate(mask_list):
    # Get the path to the segmentation
    mask_path = os.path.join(path, mask_name)
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    unique_values = np.unique(mask_array)
    for value in unique_values:
        if value != 0 and value != 255:
            print('found unexpected value: ', value)
            print(mask_name)
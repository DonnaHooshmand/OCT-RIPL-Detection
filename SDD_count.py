import os
from PIL import Image
import numpy as np

def convert_annotation(annotation_path):
    # annotation_path is the path to the annotation

    # Load the annotation
    annotation = Image.open(annotation_path)
    annotation_array = np.array(annotation)

    width, height = annotation.size

    new_array = np.zeros_like(annotation_array)

    for y in range(height):
        for x in range(width):
            # Get the pixel values at the current location
            pixel = annotation.getpixel((x, y))

            # Process the pixel values as needed
            if pixel == (0, 255, 0, 180):
                new_array[y, x] = 255
            elif pixel == (255, 0, 0, 180):
                new_array[y, x] = 125
                       
    return new_array

def find_annotation(segmentation_path, annotation_folder):
    # Find the annotation corresponding to the segmentation
    # segmentation_path is the path to the segmentation
    # annotation_folder is the folder containing the annotations

    # Get the name of the segmentation
    segmentation_name = os.path.basename(segmentation_path)

    # search for a file in the annotation folder with the same name
    # as the segmentation
    annotation_path = os.path.join(annotation_folder, segmentation_name)

    # Check if the annotation exists
    if os.path.exists(annotation_path):
        return annotation_path
    else:
        return None
    
def convert_segmentation(segmentation_path):
    # segmentation_path is the path to the segmentation

    # Load the segmentation
    segmentation = Image.open(segmentation_path)
    segmentation_array = np.array(segmentation)

    width, height = segmentation.size

    new_array = np.zeros_like(segmentation_array)

    for y in range(height):
        for x in range(width):
            # Get the pixel values at the current location
            pixel = segmentation.getpixel((x, y))
            # Process the pixel values as needed
            if pixel != (0, 0, 0, 0):
                new_array[y, x] = 255
                       
    return new_array


def get_unique_pixel_values(image):
    unique_pixel_values = set()
    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            unique_pixel_values.add(pixel)
    return unique_pixel_values

SDD_train_folder = r'C:\Users\collaborations\Desktop\SDDs\train'
SDD_val_folder = r'C:\Users\collaborations\Desktop\SDDs\val'
SDD_segmentation_folder = r'C:\Users\collaborations\Desktop\SDDs\segmentations'
target_pixel_value = (255, 0, 0, 180)
SDD_count = 0
total_count = 0


print('starting')
print(f'found {len(os.listdir(SDD_train_folder))} train files')
for filename in os.listdir(SDD_train_folder):

    total_count += 1
    print(f'file {total_count}, {filename}')
    SDD_found = False

    image_path = os.path.join(SDD_train_folder, filename)
    image = Image.open(image_path)
    print(image_path)

    # print(filename)
    # Find the annotation corresponding to the segmentation
    segmentation_file = os.path.join(SDD_segmentation_folder, filename)
    annotation_path = find_annotation(segmentation_file, SDD_train_folder)

    # Check if the annotation existss
    if annotation_path is not None:
        # Convert the annotation to a mask
        annotation_array = convert_annotation(annotation_path)
        segmentation_array = convert_segmentation(segmentation_file)
        print("segmentation_array: ", np.sum(segmentation_array, axis=2).any(), " annotation_array: ", np.sum(annotation_array, axis=2).any())
        mask_array = segmentation_array - annotation_array
        mask_array[mask_array == 130] = 255
        mask_array[mask_array == 131] = 255
        mask_array[mask_array == 1] = 0


    print("mask array sum: ", np.sum(mask_array, axis=2).any())
    if mask_array.any() > 0:
        if np.sum(mask_array, axis=2).all() <= 0:
            pass
        else: 
            print("found a thing")
            SDD_found = True
    
    annotation = False
    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            if pixel == target_pixel_value:
                annotation = True
                SDD_found = True
                # print(f"Found target pixel at ({x}, {y}) in {filename}")
    if annotation == True:
        print("annotation")
    if SDD_found == True:
        SDD_count += 1

''' # Merged two files. no need.
print(f'found {len(os.listdir(SDD_val_folder))} val files')
for filename in os.listdir(SDD_val_folder):
    # print(filename)
    
    total_count += 1
    print(f'file {total_count}')

    SDD_found = False
    image_path = os.path.join(SDD_val_folder, filename)
    image = Image.open(image_path)
    
    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            if pixel == target_pixel_value:
                SDD_found = True
                # print(f"Found target pixel at ({x}, {y}) in {filename}")
    if SDD_found == True:
        SDD_count += 1
'''
    
        
print(f'out of {total_count} B-scans, {SDD_count} had identified SDDs')
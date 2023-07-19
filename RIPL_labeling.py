from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def same_size(file1, file2):
    same_size = False
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    
    # Get the sizes of the images
    size1 = image1.size
    size2 = image2.size
    
    # Compare the sizes
    if size1 == size2:
        same_size = True
    
    # Close the images
    image1.close()
    image2.close()

    return same_size

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

def make_mask_array(segmentation_path, annotation_folder):
    # segmentation_path is the path to the segmentation
    # annotation_folder is the folder containing the annotations

    # Find the annotation corresponding to the segmentation
    annotation_path = find_annotation(segmentation_path, annotation_folder)

    # Check if the annotation exists
    if annotation_path is not None:
        # Convert the annotation to a mask
        annotation_array = convert_annotation(annotation_path)
        segmentation_array = convert_segmentation(segmentation_path)
        mask_array = segmentation_array - annotation_array
        mask_array[mask_array == 130] = 255
        mask_array[mask_array == 131] = 255
        mask_array[mask_array == 1] = 0

    else:
        # Convert the segmentation to a mask
        mask_array = convert_segmentation(segmentation_path)

    mask_array = np.sum(mask_array, axis=2)
    return mask_array

def save_mask(mask_array, mask_folder, segmentation_path):
    # mask_array is the array representing the mask
    # mask_path is the path to save
    segmentation_name = os.path.basename(segmentation_path)
    mask_path = os.path.join(mask_folder, segmentation_name)

    mask = Image.fromarray(mask_array.astype(np.uint8))

    # Save the image as a PNG file
    mask.save(mask_path)

    return 



if __name__ == "__main__":

    # Path
    segmentation_folder = "data\RIPL_data\segmentations"
    annotation_folder = "data\RIPL_data\annotations_training"
    mask_folder = "data\RIPL_data\masks"

    # Get the list of segmentations
    segmentation_list = os.listdir(segmentation_folder)

    # Loop through the segmentations
    for segmentation in segmentation_list:
        # Get the path to the segmentation
        segmentation_path = os.path.join(segmentation_folder, segmentation)

        # Check if the segmentation is a PNG file
        if segmentation.endswith(".png"):
            # Convert the segmentation to a mask
            mask_array = make_mask_array(segmentation_path, annotation_folder)

            # Save the mask
            save_mask(mask_array, mask_folder, segmentation_path)

            




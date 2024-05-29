import cv2
import numpy as np
import os


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def save_image(path, image):
    cv2.imwrite(path, image)

def combine_masks(pred_mask, annot_mask):

    final_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1]), dtype=np.uint8)

    cyan = [255, 255, 0]
    green = [0, 255, 0]
    red = [0, 0, 255]

    cyan_mask = np.all(pred_mask == cyan, axis=-1)
    green_mask = np.all(annot_mask == green, axis=-1)
    red_mask = np.all(annot_mask == red, axis=-1)

    final_mask[cyan_mask] = 255

    final_mask[green_mask] = 0

    final_mask[red_mask] = 255

    return final_mask

segmentations_dir = 'RIPL_data/segmentations'
annotations_dir = 'RIPL_data/annotations_training'
output_dir = 'ground_truth_masks'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# filename = '053020460ODYRs.png'

for filename in os.listdir(segmentations_dir):
# for i in range(1):
    pred_path = os.path.join(segmentations_dir, filename)
    annot_path = os.path.join(annotations_dir, filename)

    pred_mask = load_image(pred_path)

    if os.path.exists(annot_path):
        annot_mask = load_image(annot_path)
        final_mask = combine_masks(pred_mask, annot_mask)
    else:

        final_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1]), dtype=np.uint8)
        cyan = [255, 255, 0]
        cyan_mask = np.all(pred_mask == cyan, axis=-1)
        final_mask[cyan_mask] = 255

    output_path = os.path.join(output_dir, filename)
    save_image(output_path, final_mask)

print("Ground truth masks generated and saved in", output_dir)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import os
import pandas as pd

def count_hairs_blackhat(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    lesion_only = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    lesion_gray = cv2.cvtColor(lesion_only, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(lesion_gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat_on_lesion = cv2.bitwise_and(blackhat, blackhat, mask=mask)
    _, hair_mask = cv2.threshold(blackhat_on_lesion, 10, 255, cv2.THRESH_BINARY)
    labeled = label(hair_mask > 0)
    props = regionprops(labeled)

    filtered_hair_count = 0
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        height = maxr - minr
        width = maxc - minc
        area = prop.area
        aspect_ratio = max(height, width) / (min(height, width) + 1e-5)
        if area > 30 and aspect_ratio > 2.5:
            filtered_hair_count += 1

    return filtered_hair_count

def process_dataset(images_dir, masks_dir, output_csv):
    results = []

    for filename in os.listdir(images_dir):
        if filename.endswith(".png") and not filename.endswith("_mask.png"):
            image_path = os.path.join(images_dir, filename)
            mask_filename = filename.replace(".png", "_mask.png")
            mask_path = os.path.join(masks_dir, mask_filename)

            if os.path.exists(mask_path):
                hair_count = count_hairs_blackhat(image_path, mask_path)
                results.append({"image": filename, "hair_count": hair_count})
                print(f"Processed {filename} → Hairs: {hair_count}")
            else:
                print(f"Mask not found for {filename}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

# 🔧 Set your actual paths here
images_dir = " " #path to images'
masks_dir = " "  #path to masks 
output_csv = " " # destination path 

process_dataset(images_dir, masks_dir, output_csv)

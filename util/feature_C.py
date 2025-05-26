import cv2
import numpy as np
import os
from glob import glob
import csv
from joblib import Parallel, delayed
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
# insert paths. 
image_folder = "/Users/onealokutu/Documents/ITU/Projects in Data Science/Final Project /Dataset /images/imgs_part_1"
mask_folder = "/Users/onealokutu/Documents/ITU/Projects in Data Science/Final Project /Dataset /images/lesion_masks"
output_csv_raw = "/Users/onealokutu/Documents/ITU/Projects in Data Science/ABCD/ABC CSVs/Color.csv"

image_files = sorted([f for f in glob(os.path.join(image_folder, "*.png")) if "_mask" not in f])
mask_files = sorted([f for f in glob(os.path.join(mask_folder, "*_mask.png"))])
mask_names = {os.path.splitext(os.path.basename(f))[0].replace("_mask", ""): f for f in mask_files}
matched_images = [f for f in image_files if os.path.splitext(os.path.basename(f))[0] in mask_names]

def process_image(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = mask_names.get(base_name)
    if not mask_path:
        print(f"Mask for {base_name} not found.")
        return None

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        print(f"Error reading image or mask for {base_name}.")
        return None

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)

    reshaped = masked_image.reshape((-1, 3))
    non_black_pixels = reshaped[np.any(reshaped != [0, 0, 0], axis=1)]
    if len(non_black_pixels) == 0:
        print(f"No valid pixels in mask for {base_name}.")
        return None

    flattened_pixels = non_black_pixels.flatten()
    mean_color = np.mean(flattened_pixels)
    median_color = np.median(flattened_pixels)
    std_color = np.std(flattened_pixels)

    kmeans = MiniBatchKMeans(n_clusters=5, batch_size=500, n_init=10)
    kmeans.fit(non_black_pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    var_sum = sum(
        np.linalg.norm(d1 - d2)
        for i, d1 in enumerate(dominant_colors)
        for d2 in dominant_colors[i + 1:]
    )
    color_variation = var_sum / 10

    diversity = 0
    for i in range(5):
        if all(np.linalg.norm(dominant_colors[i] - dominant_colors[j]) >= 30 for j in range(i)):
            diversity += 1

    blue_dominant = np.sum((non_black_pixels[:, 2] > non_black_pixels[:, 0]) &
                           (non_black_pixels[:, 2] > non_black_pixels[:, 1])) / len(non_black_pixels)

    dark_ratio = np.mean(np.mean(non_black_pixels, axis=1) < 50)

    h, w = mask.shape
    left = image_rgb[:, :w // 2][mask[:, :w // 2] > 0]
    right = image_rgb[:, w // 2:][mask[:, w // 2:] > 0]
    color_asymmetry = np.linalg.norm(np.mean(left, axis=0) - np.mean(right, axis=0)) if left.size and right.size else 0

    hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV).reshape((-1, 3))
    hsv = hsv[np.any(masked_image.reshape((-1, 3)) != [0, 0, 0], axis=1)]

    hist, _ = np.histogramdd(non_black_pixels, bins=(8, 8, 8), range=((0, 256), (0, 256), (0, 256)))
    hist_norm = hist / np.sum(hist)
    entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))

    highly_sat_ratio = np.mean(hsv[:, 1] > 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)
    edge_colors = image_rgb[edges > 0]
    if edge_colors.size:
        border_color = np.mean(edge_colors, axis=0)
        lesion_color = np.mean(non_black_pixels, axis=0)
        border_contrast = np.linalg.norm(border_color - lesion_color)
    else:
        border_contrast = 0

    color_ranges = {
        'white': ((200, 200, 200), (255, 255, 255)),
        'red': ((0, 0, 150), (100, 100, 255)),
        'light_brown': ((150, 100, 50), (200, 150, 100)),
        'dark_brown': ((50, 30, 10), (100, 70, 40)),
        'blue_green': ((0, 100, 100), (50, 180, 150)),
        'black': ((0, 0, 0), (50, 50, 50))
    }

    detected_colors = []
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(image_rgb, lower, upper)
        if np.any(cv2.bitwise_and(color_mask, color_mask, mask=binary_mask)):
            detected_colors.append(color)

    color_order = ['white', 'red', 'light_brown', 'dark_brown', 'blue_green', 'black']
    color_presence = [1 if c in detected_colors else 0 for c in color_order]
    color_sum = sum(color_presence)

    return [base_name] + \
           [round(mean_color, 2), round(median_color, 2), round(std_color, 2)] + \
           [tuple(c) for c in dominant_colors] + \
           [round(color_variation, 2), diversity, round(color_asymmetry, 2),
            round(blue_dominant, 3), round(dark_ratio, 3),
            round(entropy, 3), round(highly_sat_ratio, 3),
            round(border_contrast, 2)] + color_presence + [color_sum]

header = [
    "Image",
    "Mean Color", "Median Color", "Std Color",
    "Color 1", "Color 2", "Color 3", "Color 4", "Color 5",
    "Color Variation", "Color Diversity", "Color Asymmetry",
    "Blue Dominance", "Dark Ratio",
    "Color Entropy", "Highly Sat Ratio", "Border Contrast",
    "White", "Red", "Light Brown", "Dark Brown", "Blue Green", "Black", "Color Count"
]

with open(output_csv_raw, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    results = Parallel(n_jobs=-1)(delayed(process_image)(img_path) for img_path in matched_images)
    for result in results:
        if result:
            writer.writerow(result)

print(f"\nFeature CSV saved to: {output_csv_raw}")

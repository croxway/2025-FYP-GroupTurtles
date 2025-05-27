import cv2
import numpy as np
import os
from glob import glob
import csv
from joblib import Parallel, delayed
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import time

# insert paths. 
image_folder = " "
mask_folder = " "
output_csv_raw = " "

print("Starting color feature extraction...")
print(f"Image folder: {image_folder}")
print(f"Mask folder: {mask_folder}")
print(f"Output CSV: {output_csv_raw}")

if not os.path.exists(image_folder):
    print(f"ERROR: Image folder does not exist: {image_folder}")
    exit(1)
if not os.path.exists(mask_folder):
    print(f"ERROR: Mask folder does not exist: {mask_folder}")
    exit(1)

os.makedirs(os.path.dirname(output_csv_raw), exist_ok=True)

print("\nSearching for image and mask files...")
image_files = sorted([f for f in glob(os.path.join(image_folder, "*.png")) if "_mask" not in f])
mask_files = sorted([f for f in glob(os.path.join(mask_folder, "*_mask.png"))])

print(f"Found {len(image_files)} image files")
print(f"Found {len(mask_files)} mask files")

mask_names = {os.path.splitext(os.path.basename(f))[0].replace("_mask", ""): f for f in mask_files}
matched_images = [f for f in image_files if os.path.splitext(os.path.basename(f))[0] in mask_names]

print(f"Found {len(matched_images)} matched image-mask pairs")

if len(matched_images) == 0:
    print("ERROR: No matching image-mask pairs found!")
    exit(1)

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

    # Resize mask if shape doesn't match image
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

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
           [round(color_variation, 2), diversity, round(color_asymmetry, 2),
            round(blue_dominant, 3), round(dark_ratio, 3),
            round(entropy, 3), round(highly_sat_ratio, 3),
            round(border_contrast, 2)] + color_presence + [color_sum]

header = [
    "Image",
    "Mean Color", "Median Color", "Std Color",
    "Color Variation", "Color Diversity", "Color Asymmetry",
    "Blue Dominance", "Dark Ratio",
    "Color Entropy", "Highly Sat Ratio", "Border Contrast",
    "White", "Red", "Light Brown", "Dark Brown", "Blue Green", "Black", "Color Count"
]

print(f"\nProcessing {len(matched_images)} images...")
start_time = time.time()

with open(output_csv_raw, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    
    print("Running parallel processing...")
    results = Parallel(n_jobs=-1)(delayed(process_image)(img_path) for img_path in matched_images)
    
    successful_results = 0
    failed_results = 0
    
    for result in results:
        if result:
            writer.writerow(result)
            successful_results += 1
        else:
            failed_results += 1

end_time = time.time()
processing_time = end_time - start_time

print(f"\nProcessing completed!")
print(f"Time taken: {processing_time:.2f} seconds")
print(f"Successfully processed: {successful_results} images")
print(f"Failed to process: {failed_results} images")
print(f"Output saved to: {output_csv_raw}")

if os.path.exists(output_csv_raw):
    try:
        df = pd.read_csv(output_csv_raw)
        print(f"CSV file contains {len(df)} rows and {len(df.columns)} columns")
        print("First few rows:")
        print(df.head())
    except Exception as e:
        print(f"Error reading the output CSV: {e}")
else:
    print("ERROR: Output CSV file was not created!")

print("\nDone!")

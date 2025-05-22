import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os
from glob import glob
import csv
from joblib import Parallel, delayed
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

image_folder = "imgs_part_1"
mask_folder = "lesion_masks"
output_csv_raw = os.path.join(image_folder, "color_features.csv")
output_csv_scaled = os.path.join(image_folder, "color_features_scaled.csv")

image_paths = sorted([
    f for f in glob(os.path.join(image_folder, "*.[pP][nN][gG]"))
    if "_mask" not in os.path.basename(f)
])
print(f"Found {len(image_paths)} image files.")

def process_image(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(mask_folder, f"{base_name}_mask.png")

    if not os.path.exists(mask_path):
        print(f" Mask missing for: {base_name}")
        return None

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        print(f" Failed to read image/mask: {base_name}")
        return None

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)

    reshaped = masked_image.reshape((-1, 3))
    non_black_pixels = reshaped[np.any(reshaped != [0, 0, 0], axis=1)]
    if len(non_black_pixels) == 0:
        print(f" No valid pixels in mask: {base_name}")
        return None

    kmeans = MiniBatchKMeans(n_clusters=5, batch_size=500, n_init=10)
    kmeans.fit(non_black_pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    dominant_color_strings = ['-'.join(map(str, c)) for c in dominant_colors]

    var_sum = sum(
        np.linalg.norm(d1 - d2)
        for i, d1 in enumerate(dominant_colors)
        for d2 in dominant_colors[i + 1:]
    )
    color_variation = var_sum / 10

    diversity = sum(
        all(np.linalg.norm(dominant_colors[i] - dominant_colors[j]) >= 30 for j in range(i))
        for i in range(5)
    )

    blue_dominant = np.sum((non_black_pixels[:, 2] > non_black_pixels[:, 0]) &
                           (non_black_pixels[:, 2] > non_black_pixels[:, 1])) / len(non_black_pixels)

    dark_ratio = np.mean(np.mean(non_black_pixels, axis=1) < 50)

    h, w = mask.shape
    left = image_rgb[:, :w // 2][mask[:, :w // 2] > 0]
    right = image_rgb[:, w // 2:][mask[:, w // 2:] > 0]
    color_asymmetry = np.linalg.norm(np.mean(left, axis=0) - np.mean(right, axis=0)) if left.size and right.size else 0

    mean_r, mean_g, mean_b = np.mean(non_black_pixels, axis=0)
    std_r, std_g, std_b = np.std(non_black_pixels, axis=0)

    hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV).reshape((-1, 3))
    hsv = hsv[np.any(masked_image.reshape((-1, 3)) != [0, 0, 0], axis=1)]
    hue_mean, sat_mean, val_mean = np.mean(hsv, axis=0)
    hue_std, sat_std, val_std = np.std(hsv, axis=0)

    hist, _ = np.histogramdd(non_black_pixels, bins=(8, 8, 8), range=((0, 256),) * 3)
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

    return [base_name] + dominant_color_strings + [
        round(color_variation, 2), diversity, round(color_asymmetry, 2),
        round(blue_dominant, 3), round(dark_ratio, 3),
        round(mean_r, 2), round(mean_g, 2), round(mean_b, 2),
        round(std_r, 2), round(std_g, 2), round(std_b, 2),
        round(hue_mean, 2), round(sat_mean, 2), round(val_mean, 2),
        round(hue_std, 2), round(sat_std, 2), round(val_std, 2),
        round(entropy, 3), round(highly_sat_ratio, 3),
        round(border_contrast, 2)
    ]

with open(output_csv_raw, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Image",
        "Color 1", "Color 2", "Color 3", "Color 4", "Color 5",
        "Color Variation", "Color Diversity", "Color Asymmetry",
        "Blue Dominance", "Dark Ratio",
        "Mean R", "Mean G", "Mean B",
        "Std R", "Std G", "Std B",
        "Hue Mean", "Sat Mean", "Val Mean",
        "Hue Std", "Sat Std", "Val Std",
        "Color Entropy", "Highly Sat Ratio", "Border Contrast"
    ])

    results = Parallel(n_jobs=-1)(delayed(process_image)(img_path) for img_path in image_paths)
    valid_results = [r for r in results if r]
    for row in valid_results:
        writer.writerow(row)

print(f"\n Feature CSV saved to: {output_csv_raw}")
print(f"Total valid entries: {len(valid_results)}")

df = pd.read_csv(output_csv_raw)
numeric_cols = df.columns[6:]

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df[numeric_cols] = df[numeric_cols].round(2)
df.to_csv(output_csv_scaled, index=False)
print(f" Scaled feature CSV saved to: {output_csv_scaled}")

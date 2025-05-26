import os
import subprocess
import pandas as pd
import numpy as np
import cv2
from glob import glob
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
import csv

# === File paths ===
repo_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(repo_dir, "util")

# Feature scripts (no need for feature_c_script now)
feature_a_script = "feature_a.py"
feature_b_script = "feature_b.py"
haralick_script = "haralick.py"

# CSVs
csv_a_csv = "/Users/onealokutu/Documents/ITU/Projects in Data Science/ABCD/ABC CSVs/Asymmetry.csv"
csv_b_csv = "/Users/onealokutu/Documents/ITU/Projects in Data Science/ABCD/ABC CSVs/Border.csv"
csv_haralick_csv = "/Users/onealokutu/Documents/ITU/Projects in Data Science/ABCD/ABC CSVs/Haralick_BWV.csv"

# === Input/output for color feature ===
lesion_image_folder = "/Users/onealokutu/Documents/ITU/Projects in Data Science/Lesion_only + hair removed"  # Change this to your image folder
output_csv_raw = "/Users/onealokutu/Documents/ITU/Projects in Data Science/ABCD/ABC CSVs/Color.csv"  # You control this path
lesion_images = sorted(glob(os.path.join(lesion_image_folder, "*.png")))

def process_lesion_only_image(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {base_name}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshaped = image_rgb.reshape((-1, 3))
    non_black_pixels = reshaped[np.any(reshaped != [0, 0, 0], axis=1)]
    if len(non_black_pixels) == 0:
        print(f"No valid pixels for {base_name}.")
        return None

    flattened_pixels = non_black_pixels.flatten()
    mean_color = np.mean(flattened_pixels)
    median_color = np.median(flattened_pixels)
    std_color = np.std(flattened_pixels)

    kmeans = MiniBatchKMeans(n_clusters=5, batch_size=500, n_init=10)
    kmeans.fit(non_black_pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    var_sum = sum(np.linalg.norm(d1 - d2) for i, d1 in enumerate(dominant_colors) for d2 in dominant_colors[i + 1:])
    color_variation = var_sum / 10

    diversity = sum(all(np.linalg.norm(dominant_colors[i] - dominant_colors[j]) >= 30 for j in range(i)) for i in range(5))

    blue_dominant = np.sum((non_black_pixels[:, 2] > non_black_pixels[:, 0]) &
                           (non_black_pixels[:, 2] > non_black_pixels[:, 1])) / len(non_black_pixels)
    dark_ratio = np.mean(np.mean(non_black_pixels, axis=1) < 50)

    h, w, _ = image_rgb.shape
    left = image_rgb[:, :w // 2]
    right = image_rgb[:, w // 2:]
    color_asymmetry = np.linalg.norm(np.mean(left.reshape(-1, 3), axis=0) - np.mean(right.reshape(-1, 3), axis=0))

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).reshape((-1, 3))
    hsv = hsv[np.any(image_rgb.reshape((-1, 3)) != [0, 0, 0], axis=1)]

    hist, _ = np.histogramdd(non_black_pixels, bins=(8, 8, 8), range=((0, 256), (0, 256), (0, 256)))
    hist_norm = hist / np.sum(hist)
    entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
    highly_sat_ratio = np.mean(hsv[:, 1] > 150)

    edges = cv2.Canny(image_rgb, 100, 200)
    edge_colors = image_rgb[edges > 0]
    border_contrast = np.linalg.norm(np.mean(edge_colors, axis=0) - np.mean(non_black_pixels, axis=0)) if edge_colors.size else 0

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
        color_mask = cv2.inRange(image_rgb, np.array(lower), np.array(upper))
        if np.any(color_mask):
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
    results = Parallel(n_jobs=-1)(delayed(process_lesion_only_image)(img_path) for img_path in lesion_images)
    for result in results:
        if result:
            writer.writerow(result)

print(f"\n✅ Feature CSV saved to: {output_csv_raw}")

# === Step 1: Run other feature extractors ===
def run_feature(script_name):
    result = subprocess.run(["python", script_name], cwd=util_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error running {script_name}:\n{result.stderr}")
    else:
        print(f"✅ {script_name} completed.")

# === Step 2: Merge CSVs ===
def clean_mask_name(name):
    base = os.path.splitext(name)[0]
    return base[:-5] if base.endswith("_mask") else base

def merge_csv_files(csv_files):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'key'})
        df['key'] = df['key'].apply(clean_mask_name)
        dfs.append(df)
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='key', how='inner')
    return merged_df

# === Step 3: Merge with metadata ===
def create_classification_dataset():
    baseline_features_path = os.path.join(repo_dir, "baseline_features_extended.csv")
    metadata_path = "/Users/onealokutu/Documents/ITU/Projects in Data Science/Final Project /2025-FYP-Turtles/metadata.csv"

    baseline_df = pd.read_csv(baseline_features_path)
    metadata_df = pd.read_csv(metadata_path)
    metadata_df['img_id'] = metadata_df['img_id'].str.replace(".png", "", regex=False)

    diagnosis_map = {
        'MEL': 1, 'BCC': 1, 'SCC': 1,
        'NEV': 0, 'SEB': 0, 'ACK': 0
    }
    metadata_df['label'] = metadata_df['diagnostic'].map(diagnosis_map)

    merged_df = pd.merge(baseline_df, metadata_df[['img_id', 'label']], left_on='key', right_on='img_id', how='inner')
    merged_df.drop(columns=['img_id'], inplace=True)

    output_path = os.path.join(repo_dir, "Baseline_classification_extended.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Extended classification dataset saved at {output_path}")
    return merged_df

# === Main ===
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1: GENERATING OTHER FEATURES")
    print("=" * 50)

    run_feature(feature_a_script)
    run_feature(feature_b_script)
    run_feature(haralick_script)

    print("\nSTEP 2: MERGING CSV FILES")
    merged = merge_csv_files([
        csv_a_csv,
        csv_b_csv,
        output_csv_raw,  # Color.csv we just created
        csv_haralick_csv
    ])
    output_file = os.path.join(repo_dir, "baseline_features_extended.csv")
    merged.to_csv(output_file, index=False)
    print(f"✅ Extended baseline features CSV saved at {output_file}")

    print("\nSTEP 3: CREATING EXTENDED CLASSIFICATION DATASET")
    create_classification_dataset()

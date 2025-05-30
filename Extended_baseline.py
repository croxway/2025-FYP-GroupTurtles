import os
import re
import subprocess
import pandas as pd
import numpy as np
import cv2
from glob import glob
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
import csv


repo_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(repo_dir, "util")


csv_paths = {
    "asymmetry": "" ,
    "border": " ",
    "color": " ",
    "haralick": " ",
    "hair": " "
}

output_baseline_csv = os.path.join(repo_dir, "baseline_features_extended.csv")
output_classification_csv = os.path.join(repo_dir, "Baseline_classification_extended.csv")
metadata_path = " "


lesion_image_folder = " " # path to lesion only+ hair removed
lesion_images = sorted(glob(os.path.join(lesion_image_folder, "*.png")))


def process_lesion_only_image(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    base_name = re.sub(r'_lesions?$', '', base_name, flags=re.IGNORECASE)
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshaped = image_rgb.reshape((-1, 3))
    non_black_pixels = reshaped[np.any(reshaped != [0, 0, 0], axis=1)]
    if len(non_black_pixels) < 5:
        return None  

    mean_color = np.mean(non_black_pixels)
    median_color = np.median(non_black_pixels)
    std_color = np.std(non_black_pixels)

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
           [round(color_variation, 2), diversity, round(color_asymmetry, 2),
            round(blue_dominant, 3), round(dark_ratio, 3),
            round(entropy, 3), round(highly_sat_ratio, 3),
            round(border_contrast, 2)] + color_presence + [color_sum]


def run_color_extraction():
    print("🎨 Extracting color features...")
    header = [
        "key", "Mean Color", "Median Color", "Std Color",
        "Color Variation", "Color Diversity", "Color Asymmetry",
        "Blue Dominance", "Dark Ratio", "Color Entropy",
        "Highly Sat Ratio", "Border Contrast",
        "White", "Red", "Light Brown", "Dark Brown", "Blue Green", "Black", "Color Count"
    ]

    results = Parallel(n_jobs=-1)(delayed(process_lesion_only_image)(img_path) for img_path in lesion_images)
    valid_results = [r for r in results if r]

    with open(csv_paths["color"], mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(valid_results)

    print(f"Color features saved to: {csv_paths['color']}")


def run_feature(script):
    print(f"▶️ Running {script}")
    result = subprocess.run(["python", script], cwd=util_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ {script} failed:\n{result.stderr}")
    else:
        print(f"{script} done.")


def clean_name(name):
    name = os.path.splitext(name)[0]
    return re.sub(r'_(mask|lesion[s]?|nohair)$', '', name, flags=re.IGNORECASE)


def merge_feature_csvs(csv_dict):
    dfs = []
    for name, path in csv_dict.items():
        df = pd.read_csv(path)
        df.rename(columns={df.columns[0]: 'key'}, inplace=True)
        df['key'] = df['key'].apply(clean_name)
        dfs.append(df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='key', how='inner')
    return merged_df


def add_labels_to_baseline(baseline_df):
    metadata_df = pd.read_csv(metadata_path)
    metadata_df['img_id'] = metadata_df['img_id'].str.replace(".png", "", regex=False)

    label_map = {
        'MEL': 1, 'BCC': 1, 'SCC': 1,
        'NEV': 0, 'SEB': 0, 'ACK': 0
    }
    metadata_df['label'] = metadata_df['diagnostic'].map(label_map)
    merged = pd.merge(baseline_df, metadata_df[['img_id', 'label']], left_on='key', right_on='img_id', how='inner')
    merged.drop(columns=['img_id'], inplace=True)
    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: RUNNING FEATURE EXTRACTION SCRIPTS")
    print("=" * 60)
    run_feature("feature_a.py")
    run_feature("feature_b.py")
    run_feature("haralick.py")
    run_feature("Haircount.py")
    run_color_extraction()

    print("\nSTEP 2: MERGING FEATURES")
    merged_features = merge_feature_csvs(csv_paths)
    merged_features.to_csv(output_baseline_csv, index=False)
    print(f"Saved: {output_baseline_csv}")

    print("\nSTEP 3: ADDING LABELS")
    final_df = add_labels_to_baseline(merged_features)
    final_df.to_csv(output_classification_csv, index=False)
    print(f"Final classification CSV saved: {output_classification_csv}")

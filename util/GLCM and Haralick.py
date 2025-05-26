import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from mahotas import features
from skimage.util import img_as_ubyte

# === FILE PATHS ===
csv_path = " "
image_dir = " "
mask_dir = " "

# === LOAD CSV ===
df = pd.read_csv(csv_path, delimiter=';')
print(f"🔹 Loaded {len(df)} rows from baseline CSV")

# === HARALICK FEATURE NAMES ===
haralick_names = [
    'ASM', 'Contrast', 'Correlation', 'Variance', 'IDM', 'SumAverage',
    'SumVariance', 'SumEntropy', 'Entropy', 'DifferenceVariance',
    'DifferenceEntropy', 'IMC1', 'IMC2'
]

# === FEATURE EXTRACTION FUNCTION ===
def extract_texture_features(image_path, mask_path):
    try:
        img = imread(image_path)
        mask = imread(mask_path, as_gray=True)

        # Drop alpha if RGBA
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # Convert to grayscale
        gray = rgb2gray(img)
        gray = img_as_ubyte(gray)

        if np.sum(mask) == 0:
            return None

        # Crop bounding box of mask
        coords = np.argwhere(mask > 0)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        roi = gray[y0:y1, x0:x1]
        roi = np.clip(roi, 0, 255).astype(np.uint8)

        # Skip very small lesions
        if roi.shape[0] < 5 or roi.shape[1] < 5:
            return None

        # GLCM using full 2D ROI
        glcm = graycomatrix(roi, [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        # Haralick features
        haralick_feats = features.haralick(roi)
        haralick_mean = np.nan_to_num(np.mean(haralick_feats, axis=0), nan=0.0)

        haralick_dict = {f'Haralick_{name}': val for name, val in zip(haralick_names, haralick_mean)}

        return {
            'GLCM_Contrast': contrast,
            'GLCM_Correlation': correlation,
            'GLCM_Energy': energy,
            'GLCM_Homogeneity': homogeneity,
            **haralick_dict
        }

    except Exception as e:
        print(f"❌ Error processing {os.path.basename(image_path)}: {e}")
        return None

# === LOOP THROUGH ALL ROWS ===
features_list = []

for i, row in df.iterrows():
    key = row['key']
    image_path = os.path.join(image_dir, f"{key}.png")
    mask_path = os.path.join(mask_dir, f"{key}_mask.png")

    if os.path.exists(image_path) and os.path.exists(mask_path):
        features_dict = extract_texture_features(image_path, mask_path)
        if features_dict:
            features_dict['key'] = key
            features_list.append(features_dict)
    else:
        print(f"⚠️ Missing files for {key}")

# === MERGE AND SAVE ===
df_features = pd.DataFrame(features_list)
df_combined = pd.merge(df, df_features, on='key', how='left')

# Round to 2 decimal places for Excel
float_cols = df_combined.select_dtypes(include='float').columns
df_combined[float_cols] = df_combined[float_cols].round(2)

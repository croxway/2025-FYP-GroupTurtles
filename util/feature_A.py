import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.ndimage import rotate
from math import floor, ceil
from concurrent.futures import ThreadPoolExecutor

# --- Your actual asymmetry functions below ---

def cutmask(mask):
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)
    active_cols = [i for i, val in enumerate(col_sums) if val != 0]
    active_rows = [i for i, val in enumerate(row_sums) if val != 0]
    if not active_cols or not active_rows:
        return np.zeros((1, 1), dtype=bool)
    col_min, col_max = active_cols[0], active_cols[-1]
    row_min, row_max = active_rows[0], active_rows[-1]
    return mask[row_min:row_max+1, col_min:col_max+1]

def midpoint(image):
    return image.shape[0] / 2, image.shape[1] / 2

def asymmetry(mask):
    row_mid, col_mid = midpoint(mask)
    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    if upper_half.shape[0] != lower_half.shape[0]:
        lower_half = lower_half[:upper_half.shape[0], :]
    if left_half.shape[1] != right_half.shape[1]:
        right_half = right_half[:, :left_half.shape[1]]

    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    hori_xor = np.logical_xor(upper_half, flipped_lower)
    vert_xor = np.logical_xor(left_half, flipped_right)

    hori_and = np.logical_and(upper_half, flipped_lower)
    vert_and = np.logical_and(left_half, flipped_right)

    hori_a = np.sum(hori_xor)
    vert_a = np.sum(vert_xor)
    hori_s = np.sum(hori_and)
    vert_s = np.sum(vert_and)

    hori_pct = hori_a / (hori_a + hori_s) if (hori_a + hori_s) > 0 else 1.0
    vert_pct = vert_a / (vert_a + vert_s) if (vert_a + vert_s) > 0 else 1.0

    if hori_pct <= 0.18 and vert_pct <= 0.18:
        return 1
    elif hori_pct <= 0.18 or vert_pct <= 0.18:
        return 2
    else:
        return 3

def pad_mask(mask):
    diag = int(np.ceil(np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)))
    pad_y = (diag - mask.shape[0]) // 2
    pad_x = (diag - mask.shape[1]) // 2
    return np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)

def rotation_asymmetry(mask, n=8):
    scores = []
    padded = pad_mask(mask)
    angles = np.linspace(0, 360, n, endpoint=False)
    for deg in angles:
        rotated = rotate(padded, deg, reshape=False, order=0)
        scores.append(asymmetry(rotated))
    return scores

def get_asymm_results(mask):
    mask = cutmask(mask)
    scores = rotation_asymmetry(mask)
    return min(scores), round(np.mean(scores), 4)

# --- Processing each mask file ---

def process_mask(file_name):
    try:
        # Crop filename to base without _mask.png if present
        if file_name.endswith('_mask.png'):
            base_name = file_name[:-9]  # remove '_mask.png'
        else:
            base_name = file_name
        path = os.path.join(masks_dir, file_name)
        print(f"Processing: {path}")
        mask = imread(path)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        binary = mask > 0.5
        best, mean = get_asymm_results(binary)
        return {"mask_name": base_name, "best_asymmetry_score": best, "mean_asymmetry_score": mean}
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return {"mask_name": file_name, "best_asymmetry_score": None, "mean_asymmetry_score": None, "error": str(e)}

# --- Main ---

masks_dir = " " #<- insert path to mask
output_csv = " " #<- insert path to save csv

# List all mask files in the directory (only PNGs)
all_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]

with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    results = list(executor.map(process_mask, all_files))

results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

print(f"✅ Done. Results saved to {output_csv}")

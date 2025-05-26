import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray, rgb2lab, rgb2hsv
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage import img_as_float
from skimage.morphology import binary_closing, remove_small_objects, disk


# === HARALICK HELPERS ===
def compute_entropy(glcm):
    glcm = np.where(glcm > 0, glcm, 1e-10)
    return -np.sum(glcm * np.log2(glcm))

def compute_dissimilarity(glcm):
    levels = glcm.shape[0]
    i, j = np.indices((levels, levels))
    return np.sum(glcm * np.abs(i - j))

def compute_asm(glcm):
    return np.sum(glcm ** 2)


# === TEXTURE FEATURE EXTRACTION ===
def extract_texture_features(image, mask):
    gray = rgb2gray(image)
    gray = img_as_ubyte(gray)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    patch = gray[y0:y1+1, x0:x1+1]
    patch_mask = mask[y0:y1+1, x0:x1+1]
    patch = patch * patch_mask
    if np.sum(patch_mask) < 25:
        return None
    glcm = graycomatrix(patch, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    features = {}
    for p in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features[f"Haralick_{p.capitalize()}"] = np.mean(graycoprops(glcm, p))
    entropy_vals, diss_vals, asm_vals = [], [], []
    for angle in range(glcm.shape[-1]):
        glcm_slice = glcm[:, :, 0, angle]
        entropy_vals.append(compute_entropy(glcm_slice))
        diss_vals.append(compute_dissimilarity(glcm_slice))
        asm_vals.append(compute_asm(glcm_slice))
    features["Haralick_Entropy"] = np.mean(entropy_vals)
    features["Haralick_Dissimilarity"] = np.mean(diss_vals)
    features["Haralick_ASM"] = np.mean(asm_vals)
    return features


# === BWV DETECTOR ===
class BlueWhiteVeilDetector:
    def __init__(self):
        self.lab_thresholds = {
            'L_min': 20, 'L_max': 90,
            'a_min': -25, 'a_max': 10,
            'b_min': -45, 'b_max': -5
        }

    def detect(self, image, mask):
        image = img_as_float(image)
        lab = rgb2lab(image)
        hsv = rgb2hsv(image)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        L, a, b_lab = lab[..., 0], lab[..., 1], lab[..., 2]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        lab_bwv = ((L >= 20) & (L <= 90) &
                   (a >= -25) & (a <= 10) &
                   (b_lab >= -45) & (b_lab <= -5) & mask)

        hue_blue = ((h >= 0.5) & (h <= 0.8)) | ((h >= 0.45) & (h <= 0.55))
        hsv_bwv = hue_blue & (s >= 0.1) & (s <= 0.9) & (v >= 0.15) & (v <= 0.95) & mask

        blue_dominant = (b > r) & (b > g)
        blue_gray = (b > r + 0.05) & (b > g + 0.05) & (np.abs(r - g) < 0.15)
        total = r + g + b
        blue_ratio = np.divide(b, total, out=np.zeros_like(b), where=total > 0.1)
        whitish_blue = (blue_ratio > 0.35) & (total > 0.4)
        rgb_bwv = mask & ((blue_dominant) | blue_gray | whitish_blue)

        votes = lab_bwv.astype(int) + hsv_bwv.astype(int) + rgb_bwv.astype(int)
        bwv_mask = (votes >= 1)
        bwv_mask = remove_small_objects(bwv_mask, min_size=15)
        bwv_mask = binary_closing(bwv_mask, disk(2))

        lesion_area = np.sum(mask)
        bwv_area = np.sum(bwv_mask)
        return {
            'lesion_area': lesion_area,
            'bwv_area': bwv_area,
            'bwv_pct_lesion': (bwv_area / lesion_area * 100) if lesion_area > 0 else 0
        }


# === BATCH PROCESSING & CSV EXPORT ===
def process_image_folder(folder_path, output_csv):
    detector = BlueWhiteVeilDetector()
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("⚠️ No images found in the folder.")
        return

    all_results = []
    for fname in image_files:
        image_path = os.path.join(folder_path, fname)
        try:
            image = imread(image_path)
            if image.shape[-1] == 4:
                image = image[..., :3]
            gray = rgb2gray(image)
            mask = gray > 0.05

            features = extract_texture_features(image, mask) or {}
            stats = detector.detect(image, mask)

            row = {'Filename': fname, **features, **stats}
            all_results.append(row)
            print(f"✅ Processed {fname}")
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

    if all_results:
        keys = sorted(set().union(*all_results))
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n📁 Results saved to: {output_csv}")


# === RUN SCRIPT ===
if __name__ == "__main__":
    folder = "/Users/onealokutu/Documents/ITU/Projects in Data Science/Lesion_only + hair removed"  # 🔁 UPDATE
    output_csv = "lesion_texture_analysis.csv"
    process_image_folder(folder, output_csv)

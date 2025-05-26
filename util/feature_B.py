import os
import csv
import re
from skimage import io, measure
import numpy as np

def calculate_circularity(region):
    """Calculate circularity of a region."""
    if region.perimeter == 0:
        return 0
    return 4 * np.pi * region.area / (region.perimeter ** 2)

def process_mask(image_path, area_threshold=10):
    """Process a single mask and calculate average circularity of its regions."""
    mask = io.imread(image_path, as_gray=True)
    binary_mask = mask > 0.5
    label_image = measure.label(binary_mask)
    regions = measure.regionprops(label_image)

    circularities = [
        calculate_circularity(region)
        for region in regions if region.area > area_threshold
    ]
    average_circularity = np.mean(circularities) if circularities else 0
    return average_circularity

def process_directory(directory_path, output_csv):
    """Process all mask files in a directory and write results to CSV."""
    files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    print("\n📊 Circularity Scores:")
    print("-" * 40)

    results = []

    for filename in files:
        full_path = os.path.join(directory_path, filename)
        score = process_mask(full_path)
        
        # Remove '_mask.ext' from filename
        base_name = re.sub(r'_mask\.(png|jpg|jpeg|tif|tiff)$', '', filename, flags=re.IGNORECASE)
        
        results.append([base_name, round(score, 4)])
        print(f"{filename:<30} | Circularity: {score:.4f}")

    # Write to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Average_Circularity"])
        writer.writerows(results)

    print(f"\n✅ Saved results to {output_csv}")

if __name__ == "__main__":
    directory = "/Users/onealokutu/Documents/ITU/Projects in Data Science/Final Project /Dataset /images/lesion_masks"
    output_csv = "/Users/onealokutu/Documents/ITU/Projects in Data Science/ABCD/ABC CSVs/Border.csv"
    process_directory(directory, output_csv)

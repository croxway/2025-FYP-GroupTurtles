import os
import csv
from skimage import io, measure
import numpy as np

def calculate_circularity(region):
    if region.perimeter == 0:
        return 0
    return 4 * np.pi * region.area / (region.perimeter ** 2)

def process_mask(image_path, area_threshold=10):
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

def process_directory(directory_path, output_csv="circularity_scores.csv"):
    files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    results = []

    for filename in files:
        full_path = os.path.join(directory_path, filename)
        score = process_mask(full_path)
        results.append([filename, score])
        print(f"Processed {filename} | Circularity: {score:.3f}")

    # Write to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Average_Circularity"])
        writer.writerows(results)

    print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    directory = ""  # Replace with your directory path
    process_directory(directory)

import os
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

def process_directory(directory_path):
    files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    print("\n📊 Circularity Scores:")
    print("-" * 40)
    
    for filename in files:
        full_path = os.path.join(directory_path, filename)
        score = process_mask(full_path)
        print(f"{filename:<30} | Circularity: {score:.3f}")
    
    print("\n✅ Done processing all images.\n")

if __name__ == "__main__":
    directory = ""  #  Replace with your local path
    process_directory(directory)

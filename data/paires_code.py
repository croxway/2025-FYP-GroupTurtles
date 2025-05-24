import os
import pandas as pd

def normalize_mask_name(filename):
    # Remove only '_mask.png' suffix
    if filename.endswith('_mask.png'):
        return filename[:-9]
    return filename

def find_unique_pairs(image_dir, mask_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]

    print("Sample image files:", image_files[:10])
    print("Sample mask files:", mask_files[:10])

    image_map = {os.path.splitext(f)[0]: f for f in image_files}

    mask_map = {}
    for mask_file in mask_files:
        core_name = normalize_mask_name(mask_file)
        mask_map.setdefault(core_name, []).append(mask_file)

    pairs = []
    for img_core, img_file in image_map.items():
        if img_core in mask_map and mask_map[img_core]:
            selected_mask = mask_map[img_core][0]
            pairs.append({
                "image_name": img_file,
                "mask_name": selected_mask
            })

    return pairs

def save_pairs_to_excel(pairs, output_excel):
    df = pd.DataFrame(pairs)
    df.to_excel(output_excel, index=False)
    print(f"✅ Saved {len(df)} unique pairs to Excel: {output_excel}")

if __name__ == "__main__":
    image_dir = r"C:/Users/ASUS/OneDrive/Pulpit/imgs_part_1"
    mask_dir = r"C:/Users/ASUS/OneDrive/Pulpit/Projects/padchest_lesion_masks/lesion_masks"
    output_excel = r"C:/Users/ASUS/OneDrive/Pulpit/paires.xlsx"

    pairs = find_unique_pairs(image_dir, mask_dir)
    print(f"✅ Found {len(pairs)} unique image-mask pairs (one mask per image).")
    save_pairs_to_excel(pairs, output_excel)

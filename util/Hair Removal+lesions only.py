import os
import cv2
import numpy as np

def remove_hairs(image):
    """
    Removes hair from the input image using morphological operations and inpainting.

    Args:
        image (np.ndarray): Original BGR image.

    Returns:
        np.ndarray: Hair-removed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Black-hat filtering to find hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create mask of hair
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint hair
    inpainted = cv2.inpaint(image, hair_mask, 1, cv2.INPAINT_TELEA)

    return inpainted

def extract_cropped_lesions(image_folder, mask_folder, output_folder):
    """
    Extract and save cropped lesion areas from images using their corresponding masks.

    Args:
        image_folder (str): Path to folder with original skin lesion images.
        mask_folder (str): Path to folder with masks (ending in '_mask.png').
        output_folder (str): Folder to save cropped lesion images.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_filename in image_files:
        base_name, ext = os.path.splitext(image_filename)
        mask_filename = f"{base_name}_mask.png"

        image_path = os.path.join(image_folder, image_filename)
        mask_path  = os.path.join(mask_folder, mask_filename)

        if not os.path.exists(mask_path):
            print(f"⚠️ No mask found for {image_filename}, skipping.")
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"❌ Failed to load {image_filename} or its mask.")
            continue

        # Step 1: Remove hair before masking
        image_no_hair = remove_hairs(image)

        # Step 2: Apply lesion mask
        masked_image = cv2.bitwise_and(image_no_hair, image_no_hair, mask=mask)

        # Step 3: Crop the lesion
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            print(f"⚠️ Empty mask for {image_filename}, skipping.")
            continue

        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)

        cropped_lesion = masked_image[y_min:y_max+1, x_min:x_max+1]

        # Step 4: Save cropped lesion image
        save_path = os.path.join(output_folder, f"{base_name}_lesion.png")
        cv2.imwrite(save_path, cropped_lesion)
        print(f"✅ Saved: {save_path}")

# Example usage:
extract_cropped_lesions(
    image_folder="/Users/onealokutu/Documents/ITU/Projects in Data Science/Final Project /Dataset /images/imgs_part_1",
    mask_folder="/Users/onealokutu/Documents/ITU/Projects in Data Science/Final Project /Dataset /images/lesion_masks",
    output_folder="/Users/onealokutu/Documents/ITU/Projects in Data Science/Final Project /2025-FYP-Turtles/Lesion_only + hair removed"
)

import os
import cv2
import numpy as np

def remove_hairs(image):
    """
    Removes hair from the input image using morphological operations and inpainting.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, hair_mask, 1, cv2.INPAINT_TELEA)
    return inpainted

def extract_cropped_lesions(image_folder, mask_folder, output_folder):
    """
    Extracts and saves cropped lesion areas from images using corresponding masks,
    skips images with missing or invalid masks.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_filename in image_files:
        try:
            base_name, _ = os.path.splitext(image_filename)
            mask_filename = f"{base_name}_mask.png"

            image_path = os.path.join(image_folder, image_filename)
            mask_path  = os.path.join(mask_folder, mask_filename)

            if not os.path.exists(mask_path):
                print(f"⚠️ No mask found for {image_filename}, skipping.")
                continue

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"❌ Could not read {image_filename} or its mask. Skipping.")
                continue

            if mask.shape != image.shape[:2]:
                print(f"⚠️ Size mismatch for {image_filename}, resizing mask.")
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            # Remove hair
            image_no_hair = remove_hairs(image)

            # Apply lesion mask
            masked_image = cv2.bitwise_and(image_no_hair, image_no_hair, mask=mask)

            # Find bounding box
            ys, xs = np.where(mask > 0)
            if ys.size == 0 or xs.size == 0:
                print(f"⚠️ Mask is empty for {image_filename}, skipping.")
                continue

            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)

            cropped_lesion = masked_image[y_min:y_max+1, x_min:x_max+1]

            save_path = os.path.join(output_folder, f"{base_name}_lesion.png")
            cv2.imwrite(save_path, cropped_lesion)
            print(f"✅ Saved: {save_path}")

        except Exception as e:
            print(f"❌ Error processing {image_filename}: {str(e)}. Skipping.")

# Example usage:
extract_cropped_lesions(
    image_folder= "",
    mask_folder=" ",
    output_folder=" "
)

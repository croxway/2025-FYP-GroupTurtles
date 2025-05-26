import numpy as np
from skimage.color import rgb2lab, rgb2hsv, rgb2gray
from skimage.io import imread
from skimage import img_as_float
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects, binary_erosion, binary_dilation
from skimage.filters import gaussian, threshold_otsu, threshold_multiotsu
from skimage.segmentation import felzenszwalb
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt

class LesionFocusedBlueWhiteVeilDetector:
    def __init__(self):
        # Blue-white veil detection thresholds
        self.lab_thresholds = {
            'L_min': 20, 'L_max': 90,
            'a_min': -25, 'a_max': 10,
            'b_min': -45, 'b_max': -5
        }
        
        self.hsv_thresholds = {
            'h_min': 0.5, 'h_max': 0.8,
            's_min': 0.1, 's_max': 0.9,
            'v_min': 0.15, 'v_max': 0.95
        }
        
        self.min_region_size = 25
        self.morph_disk_size = 2
    
    def segment_lesion(self, image):
        """Segment the lesion from background skin"""
        # Convert to grayscale for lesion segmentation
        gray = rgb2gray(image)
        
        # Use multi-level Otsu thresholding to separate lesion from skin
        try:
            thresholds = threshold_multiotsu(gray, classes=3)
            # Take the darkest regions (lesion is typically darker than normal skin)
            lesion_mask = gray < thresholds[0]
        except:
            # Fallback to simple Otsu
            threshold = threshold_otsu(gray)
            lesion_mask = gray < threshold * 0.8  # Slightly more sensitive
        
        # Clean up the lesion mask
        lesion_mask = remove_small_objects(lesion_mask, min_size=500)
        lesion_mask = binary_closing(lesion_mask, disk(5))
        lesion_mask = binary_opening(lesion_mask, disk(3))
        
        # Fill holes and get the largest connected component (main lesion)
        labeled = label(lesion_mask)
        if labeled.max() > 0:
            # Get the largest region
            regions = regionprops(labeled)
            largest_region = max(regions, key=lambda r: r.area)
            lesion_mask = (labeled == largest_region.label)
        
        return lesion_mask
    
    def detect_blue_white_in_lesion(self, image, lesion_mask):
        """Detect blue-white veil specifically within the lesion area"""
        # Only analyze pixels within the lesion
        lesion_pixels = image[lesion_mask]
        lesion_coords = np.where(lesion_mask)
        
        if len(lesion_pixels) == 0:
            return np.zeros_like(lesion_mask), {}
        
        # Convert lesion pixels to different color spaces
        lab_image = rgb2lab(image)
        hsv_image = rgb2hsv(image)
        
        # Initialize blue-white veil mask
        bwv_mask = np.zeros_like(lesion_mask)
        
        # LAB detection within lesion
        L = lab_image[:,:,0]
        a = lab_image[:,:,1] 
        b = lab_image[:,:,2]
        
        lab_bwv = (
            (L >= self.lab_thresholds['L_min']) & 
            (L <= self.lab_thresholds['L_max']) &
            (a >= self.lab_thresholds['a_min']) & 
            (a <= self.lab_thresholds['a_max']) &
            (b >= self.lab_thresholds['b_min']) & 
            (b <= self.lab_thresholds['b_max']) &
            lesion_mask  # Only within lesion
        )
        
        # HSV detection within lesion
        h = hsv_image[:,:,0]
        s = hsv_image[:,:,1]
        v = hsv_image[:,:,2]
        
        # Broader blue hue detection
        hue_blue = ((h >= 0.5) & (h <= 0.8)) | ((h >= 0.45) & (h <= 0.55))
        hsv_bwv = (
            hue_blue &
            (s >= 0.1) & (s <= 0.9) &
            (v >= 0.15) & (v <= 0.95) &
            lesion_mask
        )
        
        # RGB detection within lesion
        r, g, b_channel = image[:,:,0], image[:,:,1], image[:,:,2]
        
        # Multiple RGB criteria for blue-white detection
        blue_dominant = (b_channel > r) & (b_channel > g)
        blue_sufficient = b_channel > 0.2
        
        # Blue-gray detection (common in blue-white veils)
        blue_gray = (
            (b_channel > r + 0.05) & 
            (b_channel > g + 0.05) & 
            (np.abs(r - g) < 0.15)
        )
        
        # Whitish-blue detection
        total_intensity = r + g + b_channel
        blue_ratio = np.divide(b_channel, total_intensity, 
                              out=np.zeros_like(b_channel), 
                              where=total_intensity>0.1)
        whitish_blue = (blue_ratio > 0.35) & (total_intensity > 0.4)
        
        rgb_bwv = lesion_mask & (
            (blue_dominant & blue_sufficient) |
            blue_gray |
            whitish_blue
        )
        
        # Combine methods with voting
        combined_votes = lab_bwv.astype(int) + hsv_bwv.astype(int) + rgb_bwv.astype(int)
        
        # Accept if at least 1 method strongly agrees (more sensitive)
        bwv_mask = combined_votes >= 1
        
        # Clean up the mask
        bwv_mask = remove_small_objects(bwv_mask, min_size=15)
        bwv_mask = binary_closing(bwv_mask, disk(2))
        
        # Calculate statistics
        lesion_area = np.sum(lesion_mask)
        bwv_area = np.sum(bwv_mask)
        
        statistics = {
            'lesion_area_pixels': lesion_area,
            'bwv_area_pixels': bwv_area,
            'bwv_percentage_of_lesion': (bwv_area / lesion_area * 100) if lesion_area > 0 else 0,
            'bwv_percentage_of_image': (bwv_area / (image.shape[0] * image.shape[1]) * 100),
            'lab_detections': np.sum(lab_bwv),
            'hsv_detections': np.sum(hsv_bwv), 
            'rgb_detections': np.sum(rgb_bwv),
            'method_agreement': {
                'lab_only': np.sum((combined_votes == 1) & lab_bwv),
                'hsv_only': np.sum((combined_votes == 1) & hsv_bwv),
                'rgb_only': np.sum((combined_votes == 1) & rgb_bwv),
                'two_methods': np.sum(combined_votes == 2),
                'all_methods': np.sum(combined_votes == 3)
            }
        }
        
        return bwv_mask, statistics, {
            'lesion_mask': lesion_mask,
            'lab_bwv': lab_bwv,
            'hsv_bwv': hsv_bwv,
            'rgb_bwv': rgb_bwv,
            'combined_votes': combined_votes
        }
    
    def analyze_image(self, image_path, visualize=True):
        """Main analysis function"""
        # Load image
        image = imread(image_path)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        
        image = img_as_float(image)
        
        # Gentle preprocessing
        image_smooth = gaussian(image, sigma=0.5, channel_axis=-1)
        
        # Segment lesion
        print("Segmenting lesion...")
        lesion_mask = self.segment_lesion(image_smooth)
        
        # Detect blue-white veil within lesion
        print("Detecting blue-white veil...")
        bwv_mask, statistics, debug_masks = self.detect_blue_white_in_lesion(image_smooth, lesion_mask)
        
        # Prepare results
        results = {
            'original_image': image_smooth,
            'lesion_mask': lesion_mask,
            'bwv_mask': bwv_mask,
            'statistics': statistics,
            'debug_masks': debug_masks
        }
        
        if visualize:
            self.visualize_results(results)
        
        return results
    
    def visualize_results(self, results):
        """Visualize the detection results"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Row 1: Original, lesion segmentation, and individual method results
        axes[0, 0].imshow(results['original_image'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(results['lesion_mask'], cmap='gray')
        axes[0, 1].set_title(f'Lesion Segmentation\n{results["statistics"]["lesion_area_pixels"]} pixels')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(results['debug_masks']['lab_bwv'], cmap='Reds', alpha=0.8)
        axes[0, 2].set_title(f'LAB Detection\n{results["statistics"]["lab_detections"]} pixels')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(results['debug_masks']['hsv_bwv'], cmap='Blues', alpha=0.8)
        axes[0, 3].set_title(f'HSV Detection\n{results["statistics"]["hsv_detections"]} pixels')
        axes[0, 3].axis('off')
        
        # Row 2: RGB detection, voting, final result, and overlay
        axes[1, 0].imshow(results['debug_masks']['rgb_bwv'], cmap='Greens', alpha=0.8)
        axes[1, 0].set_title(f'RGB Detection\n{results["statistics"]["rgb_detections"]} pixels')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(results['debug_masks']['combined_votes'], cmap='viridis', vmin=0, vmax=3)
        axes[1, 1].set_title('Method Agreement\n(0=none, 3=all)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(results['bwv_mask'], cmap='hot')
        axes[1, 2].set_title(f'Final BWV Detection\n{results["statistics"]["bwv_area_pixels"]} pixels')
        axes[1, 2].axis('off')
        
        # Overlay on original
        overlay = results['original_image'].copy()
        overlay[results['bwv_mask']] = [1, 0, 0]  # Red for BWV
        overlay[results['lesion_mask'] & ~results['bwv_mask']] = [0, 0.3, 0]  # Subtle green for lesion
        axes[1, 3].imshow(overlay)
        axes[1, 3].set_title(f'Clinical Overlay\nBWV: {results["statistics"]["bwv_percentage_of_lesion"]:.1f}% of lesion')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_bwv_score(self, statistics):
        """Get normalized blue-white veil score (0-1)"""
        return statistics['bwv_percentage_of_lesion'] / 100.0
    
    def print_clinical_assessment(self, statistics):
        """Print simple clinical assessment"""
        bwv_score = self.get_bwv_score(statistics)
        
        print(f"\n🔵 BLUE-WHITE VEIL SCORE: {bwv_score:.3f}")
        print(f"📊 LESION COVERAGE: {statistics['bwv_percentage_of_lesion']:.1f}%")
        
        if bwv_score >= 0.75:
            print("⚠️  CLINICAL RISK: VERY HIGH")
        elif bwv_score >= 0.5:
            print("⚠️  CLINICAL RISK: HIGH") 
        elif bwv_score >= 0.25:
            print("⚠️  CLINICAL RISK: MODERATE")
        elif bwv_score >= 0.1:
            print("⚠️  CLINICAL RISK: LOW")
        else:
            print("✅ CLINICAL RISK: MINIMAL")

# Usage
if __name__ == "__main__":
    detector = LesionFocusedBlueWhiteVeilDetector()
    image_path = r"C:\Users\suraj\OneDrive\Desktop\projects in d.s\imgs_part_1\PAT_101_1041_658.png"
    
    results = detector.analyze_image(image_path, visualize=True)
    detector.print_clinical_assessment(results['statistics'])

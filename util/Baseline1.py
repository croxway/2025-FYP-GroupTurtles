import numpy as np
from skimage.segmentation import slic
from skimage.color import rgba2rgb
from skimage.measure import perimeter
from scipy.ndimage import rotate
from math import floor, ceil

###############
###ASYMMETRY###
###############

def cutmask(mask):
    '''Cut empty space from mask array such that it has smallest possible dimensions.

    Args:
        mask (numpy.ndarray): mask to cut

    Returns:
        cut_mask (numpy.ndarray): cut mask
    '''
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_mask = mask[row_min:row_max+1, col_min:col_max+1]
    
    return cut_mask

def midpoint(image):
    '''Find midpoint of image array.'''
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid

def asymmetry(mask):
    '''Calculate asymmetry score and categorize between 1 and 3.'''
    row_mid, col_mid = midpoint(mask)

    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    # Ensure the halves are of the same size for proper comparison
    if upper_half.shape[0] != lower_half.shape[0]:
        lower_half = lower_half[:upper_half.shape[0], :]
    if left_half.shape[1] != right_half.shape[1]:
        right_half = right_half[:, :left_half.shape[1]]

    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    hori_and_area = np.logical_and(upper_half, flipped_lower)
    vert_and_area = np.logical_and(left_half, flipped_right)

    hori_symmetry_pxls = np.sum(hori_and_area)
    vert_symmetry_pxls = np.sum(vert_and_area)

    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    # Asymmetry percentages
    hori_asymmetry_percentage = hori_asymmetry_pxls / (hori_asymmetry_pxls + hori_symmetry_pxls)
    vert_asymmetry_percentage = vert_asymmetry_pxls / (vert_asymmetry_pxls + vert_symmetry_pxls)

    # Determine scores based on asymmetry percentages
    margin = 0.18
    hori_symmetric = hori_asymmetry_percentage <= margin
    vert_symmetric = vert_asymmetry_percentage <= margin

    if hori_symmetric and vert_symmetric:
        return 1
    elif hori_symmetric or vert_symmetric:
        return 2
    else:
        return 3

def pad_mask(mask):
    '''Pad the mask to avoid out-of-bound areas during rotation.'''
    diagonal = int(np.ceil(np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)))
    padded_size = (diagonal, diagonal)
    pad_y = (padded_size[0] - mask.shape[0]) // 2
    pad_x = (padded_size[1] - mask.shape[1]) // 2
    return np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)

def rotation_asymmetry(mask, n):
    '''Rotate mask n times and calculate asymmetry score for each iteration.'''
    asymmetry_scores = {}
    padded_mask = pad_mask(mask)

    for i in range(n):
        degrees = 360 * i / n
        rotated_mask = rotate(padded_mask, degrees, reshape=False)
        score = asymmetry(rotated_mask)
        asymmetry_scores[degrees] = score

    return asymmetry_scores

def best_asymmetry(mask, rotations=30):
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    return min(asymmetry_scores.values())

def mean_asymmetry(mask, rotations=30):
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    return sum(asymmetry_scores.values()) / len(asymmetry_scores)


def get_asymm_results(mask):
    mask = cutmask(mask)
    best = best_asymmetry(mask)
    mean = mean_asymmetry(mask)
    return [best,mean]

####################
### BORDER (B) #####
####################

def border_irregularity(mask):
    '''Categorize border irregularity into 1 (regular) to 3 (irregular).'''
    mask = (mask > 0).astype(np.uint8)
    area = np.sum(mask)
    peri = perimeter(mask, neighborhood=8)

    if area == 0:
        return 1  # default to "regular" if area is 0

    index = (peri**2) / (4 * np.pi * area)

    if index <= 1.2:
        return 1
    elif index <= 1.8:
        return 2
    else:
        return 3

###########
###COLOR###
###########

def get_rgb_means(image, slic_segments):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original RGB image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        rgb_means (list): RGB mean values for each segment.
    '''

    # Get the maximum segment ID
    max_segment_id = np.max(slic_segments)

    # Initialize list to store RGB mean values for each segment
    rgb_means = []

    # Iterate over each segment
    for i in range(1, max_segment_id + 1):

        # Create a masked image where only the pixels belonging to the current segment are retained
        segment = image.copy()
        segment = segment.astype(float)
        segment[slic_segments != i] = np.nan
        
        

        # Compute mean RGB values
        rgb_mean = np.nanmean(segment, axis=(0, 1))

        # Append the mean RGB values to the list
        rgb_means.append(rgb_mean)

    return rgb_means

colors = {"red1": (100, 34, 50),
              "red2" : (173, 132, 132),
              "dark brown" : (92,64,51),
              "light brown" : (160, 120, 90),
              "blue gray" : (112,119,163),
              "white" : (200,200,200),
              "black" : (50, 50, 50)
              }
def classify(rgb_tuple):
    # Define a lambda function for calculating the Manhattan distance between two RGB tuples.
    manhattan = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])

    # Calculate the Manhattan distance between the input RGB tuple and each color in `colors`.
    distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}

    # Initialize an empty list to hold the names of the colors that are close to the input RGB tuple.
    color = [key for key, value in distances.items() if value < 50]

    # Add additional colors to the list if their distance is less than 90 and they are specific colors.
    color += [key for key, value in distances.items() if value < 90 and key in ["black", "red1", "red2", "white"]]

    # Return the list of color names.
    return color

def find_colors(image, mask):
        color_dict = {}

        # Convert RGBA image to RGB
        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        # Apply SLIC segmentation
        segments = slic(image, start_label=1, mask=mask, n_segments=50)  # Adjust n_segments as needed

        colors_in_lesion = []

        for meanrgb in get_rgb_means(image, segments):
            r, g, b = meanrgb
            if r < 1 and g < 1 and b < 1:
                correctedrgb = (r * 255,g * 255, b * 255)
            else:
                correctedrgb = r,g,b
            for color in classify(correctedrgb):
                colors_in_lesion.append(color)

        colors_in_lesion = set(colors_in_lesion)
        for i in ["red1","red2","white","black","light brown", "dark brown", "blue gray"]:
            color_dict[i] = int(i in colors_in_lesion)

        return [i for i in color_dict.values()]


def extract_features(image, mask):
    features_imagex = get_asymm_results(mask)
    features_imagex.append(border_irregularity(mask))
    features_imagex += find_colors(image, mask)
    return np.array(features_imagex, dtype=np.float16)
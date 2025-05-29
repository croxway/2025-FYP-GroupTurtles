![image](https://github.com/user-attachments/assets/74bbf5df-842a-4e69-a015-592d85610e94)# Projects in Data Science (2025)

# 1. Exploration of the Data/ Summary of the Data:

Dataset Overview
The dataset consists of 200 images of skin moulds that as used by GroupH in the mandatory project at the start of the semester.

Each image has been annotated with a hair presence rating:

0 = No hair

1 = Some hair

2 = A lot of hair

The primary goal is to develop an algorithm to remove hair from images, enabling better differentiation between cancerous and non-cancerous moulds.

 #  Summary Statistics:
The group uniformly agreed on the hair annotation for 119 pictures (59.5%), with an average agreement of 89.0%. The lowest agreement was 20% for the picture img_0925.png.

# Image Quality & Variability:

The dataset contains images of varying resolutions, lighting conditions, and skin tones.

Some images have darker or lighter backgrounds, which might impact segmentation models.

Hair thickness and density vary across images with ratings 1 and 2.

# Potential Challenges:

Hair thickness variation: Some images may contain thin, scattered hair, while others have dense, overlapping hair.

Skin tone variations: The effectiveness of hair removal may differ for lighter vs. darker skin tones.

Low-contrast images: Some moulds blend into the skin, making segmentation harder.

# 2. Annotation of the Picture:
   
As a group we agreed to rate the various datasets based on the number of hairs that were present on the image and not solely on the lesions.
   
 Below are some examples of data were we collectively agreed had the same ratings:
   
  ![img_0917](https://github.com/user-attachments/assets/3e886e7e-c603-42fd-8662-dd8d6a1d2aa7)
 - We agreed that this image had a rating of 1 due to the presence of some hair.
   
 ![img_0918](https://github.com/user-attachments/assets/73e4147b-0fce-45ba-9e3a-ed223c744dec)
 - We gave this a rating of 2 because it contains a lot of hair.
  
 ![img_0919](https://github.com/user-attachments/assets/6992880a-4b9e-438c-9d20-fd3b4082a3ce)
 - A rating of 0 was given because there was no hair found.

##  Disagreement:
   
![img_0925](https://github.com/user-attachments/assets/90367a09-a4d6-4bf7-8e68-04e372d1718c)
 
-  With this particular image, the group had 3 different ratings on the image. As mentioned above, there was an agreement to rate the number of hairs present in the image but clearly some of us also focused on the image present on the lesion and therefore disregarded the hair present which we assumed to be the persons actual hair. 


# 4. Segmentation of the Hair
   ## First Test Case
   
   Original Picture:
   
   ![image](https://github.com/user-attachments/assets/989d4686-10df-409e-be98-bcfd3edd003b)

   Modified Picture:
   
   ![image](https://github.com/user-attachments/assets/c19813cf-3c6c-47dc-a315-5be02f060c22)

- Tested the hair removal function with this picture.
- The original image had significant hair and was rated a two by all group members.
- The function effectively removed dark hair but struggled with white hair.
- A drawback is the removal of darker spots on the mole, which may affect color and shape analysis.

## Some issues with the hair removal code

For this image, we all agreed that it had no hair, but due to some skin condition, the hair removal code altered it anyway, significantly affecting the lesion itself and possibly making it impossible to accurately diagnose.
![merged_919](https://github.com/user-attachments/assets/3928b0bd-29dd-45ed-bed8-21e4b45a76da)

Also another case where it actually had a lot of hair (rating 2 by all of us). We can see that probably because the hair has similar colour with the lesion, it doesn't really remove the hair on top of it, and also seems to fuse them together.
![merged_1116](https://github.com/user-attachments/assets/39fd7d4e-7609-4ce3-80a7-d4f665c614eb)

## Experimenting with different Kernel, Threshold and Radius sizes

For checking different effects of the parameters, we chose the picture img_0918 because it has a lot of hair to be removed, shows dark and white hair, and different colored spots on the lesion:

<img width="192" alt="image" src="https://github.com/user-attachments/assets/bb03cf18-8176-4b92-8429-e352ee725484" />

### Kernel Size

| **Kernel Size 10**                                   | **Kernel Size 25 (original)**                                   | **Kernel Size 40**                                   |
|:-----------------------------------------------------:|:-----------------------------------------------------:|:-----------------------------------------------------:|
| <img width="192" alt="image" src="https://github.com/user-attachments/assets/0e67dc11-7938-45a9-9106-0b8f837efdc9" /> | <img width="192" alt="image" src="https://github.com/user-attachments/assets/48be8d7d-ffc6-4d77-b4e4-4af21fdffbda" /> | <img width="192" alt="image" src="https://github.com/user-attachments/assets/c49434dc-2562-46bb-b4b3-7ff48359edd4" /> |
| - Less hair removal, showing more skin texture (could be better for checking lesion colors). | - Balanced hair removal with some smoothing, but it doesnt show some lesion color details. | - Strong hair removal, but white hair isn’t cleared well and it blurs the surroundings. |

### Threshold

| **Threshold 5**                                   | **Threshold 10 (original)**                                   | **Threshold 15**                                   |
|:-------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------:|
| <img width="192" alt="image" src="https://github.com/user-attachments/assets/7373eb6e-d9c4-4d14-b3f5-1069aeed0790" /> | <img width="192" alt="image" src="https://github.com/user-attachments/assets/d67a49d0-517f-4fe8-8a2a-91566d45b097" /> | <img width="192" alt="image" src="https://github.com/user-attachments/assets/781fe832-cc64-47af-9f06-79b12684b922" /> |
| - Catches more hair (though almost no white hair). <br> - The image gets a bit noisy and loses some lesion details. | - Works balanced with good hair removal. | - Removes up less hair; some gray hair stays, but more lesion details are kept. |

### Radius

| **Radius 1**                                   | **Radius 3 (original)**                                   | **Radius 5**                                   |
|:-------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------:|
| <img width="192" alt="image" src="https://github.com/user-attachments/assets/55348ff6-770a-4bb4-bc77-d16ce98a7212" /> | <img width="192" alt="image" src="https://github.com/user-attachments/assets/f4ebe0b6-ff83-4a18-b116-2f4fd10ff670" /> | <img width="192" alt="image" src="https://github.com/user-attachments/assets/834ecb25-e44e-4765-9b0c-de4c2252ae5d" /> |
| - Small fill area, keeping smaller details. <br> - May leave tiny spots where hair was removed. | - A good balance; fills the area while keeping details. | - Bigger fill area for smoother results. <br> - Might lose some fine lesion details. |


*Conclusion
This project gave us valuable insights into the challenges of hair removal in medical imaging. We found that image variability—such as hair color, skin tone, and lighting—can significantly affect the performance of our algorithm. While adjusting parameters like kernel size, threshold, and radius helped improve results in some cases, it also highlighted the limitations of a one-size-fits-all approach. Going forward, a more adaptive or learning-based method may be necessary to consistently preserve lesion details while effectively removing hair. These findings will help guide our approach in the final assignment, where the goal will be to build a more robust and generalizable solution.

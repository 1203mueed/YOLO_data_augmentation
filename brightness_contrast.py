import os
import shutil
import numpy as np
import cv2
import random

# Paths
original_images_folder = 'data/images'
modified_images_folder = 'brightness_contrast/images'
annotations_folder = 'data/labels'
modified_annotations_folder = 'brightness_contrast/labels'
os.makedirs(modified_images_folder, exist_ok=True)
os.makedirs(modified_annotations_folder, exist_ok=True)

# Get a list of all image files
image_files = [f for f in os.listdir(original_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Randomly select 25% of images
num_images_to_modify = int(0.25 * len(image_files))
images_to_modify = random.sample(image_files, num_images_to_modify)

# Process each selected image
for image_file in images_to_modify:
    image_path = os.path.join(original_images_folder, image_file)
    image = cv2.imread(image_path)

    # Adjust brightness and contrast (you can modify these values)
    brightness = -20  # Example: increase brightness by 50
    contrast = 1.75  # Example: increase contrast by 20%
    modified_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2HSV)

    # Increase saturation by a given value (e.g., 20)
    saturation_value = 5
    hsv_image[..., 1] = cv2.add(hsv_image[..., 1], saturation_value)

    # Clip resulting values to fit within 0 - 255 range
    hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)

    # Convert back to BGR color space
    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Save the modified image
    modified_image_name = os.path.splitext(image_file)[0] + '_brightness.jpg'
    modified_image_file = os.path.join(modified_images_folder, modified_image_name)
    cv2.imwrite(modified_image_file, modified_image)

    # Copy the original annotation to the new folder
    annotation_file = os.path.join(annotations_folder, image_file.replace('.jpg', '.txt'))
    modified_annotation_file = os.path.join(modified_annotations_folder, image_file.replace('.jpg', '_brightness.txt'))
    shutil.copyfile(annotation_file, modified_annotation_file)

print(f"{num_images_to_modify} images modified and saved successfully!")

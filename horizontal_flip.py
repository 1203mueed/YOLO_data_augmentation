import os
import random
import cv2

# Paths
original_images_folder = 'data/images'
flipped_images_folder = 'horizontally_flipped/images'
os.makedirs(flipped_images_folder, exist_ok=True)

# Get a list of all image files
image_files = [f for f in os.listdir(original_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Randomly select 25% of images
num_images_to_flip = int(0.25 * len(image_files))
images_to_flip = random.sample(image_files, num_images_to_flip)

# Process each selected image
for image_file in images_to_flip:
    image_path = os.path.join(original_images_folder, image_file)
    image = cv2.imread(image_path)

    # Horizontally flip the image
    flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip

    # Save the flipped image with a new name
    flipped_image_name = os.path.splitext(image_file)[0] + '_horiz.jpg'
    flipped_image_file = os.path.join(flipped_images_folder, flipped_image_name)
    cv2.imwrite(flipped_image_file, flipped_image)

print(f"{num_images_to_flip} images flipped and saved successfully with new names!")


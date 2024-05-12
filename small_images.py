import os
from PIL import Image
import numpy as np

# Function to load YOLO annotations and convert them to bounding box coordinates
def load_annotations(filename, img_width, img_height):
    with open(filename, 'r') as file:
        boxes = []
        for line in file.readlines():
            class_id, cx, cy, bw, bh = map(float, line.split())
            # Convert from normalized to absolute coordinates
            x1 = (cx - bw / 2) * img_width
            y1 = (cy - bh / 2) * img_height
            x2 = (cx + bw / 2) * img_width
            y2 = (cy + bh / 2) * img_height
            boxes.append((class_id, x1, y1, x2, y2))
    return boxes

# Function to save new YOLO annotations for cropped images
def save_normalized_annotations(filename, class_id):
    with open(filename, 'w') as file:
        # Since the cropped image represents the entire object, the bounding box
        # will cover the whole image. Hence, the normalized values are fixed.
        file.write(f"{int(class_id)} 0.5 0.5 1 1\n")

# Create directories for the cropped images and annotations
os.makedirs('cropped_images', exist_ok=True)
os.makedirs('cropped_annotations', exist_ok=True)

# List all images and shuffle them to process a random subset
image_files = os.listdir('data/images')
np.random.shuffle(image_files)
subset_size = int(len(image_files) * 0.25)

for img_file in image_files[:subset_size]:
    img_path = f'data/images/{img_file}'
    img = Image.open(img_path)
    img_width, img_height = img.size
    ann_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
    boxes = load_annotations(f'data/labels/{ann_file}', img_width, img_height)

    for box in boxes:
        class_id, x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        image_area = img_width * img_height

        # Check if the bounding box area is less than 15% of the whole image
        if box_area < image_area * 0.15:
            # Crop the image to the bounding box
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img_path = f'cropped_images/{img_file[:-4]}_{int(class_id)}_cropped.jpg'
            cropped_img.save(cropped_img_path)

            # Save the new normalized annotation file
            cropped_ann_path = f'cropped_annotations/{img_file[:-4]}_{int(class_id)}_cropped.txt'
            save_normalized_annotations(cropped_ann_path, class_id)

print("Cropping and annotation normalization complete!")

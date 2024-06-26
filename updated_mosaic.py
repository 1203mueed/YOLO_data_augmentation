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

# Function to save new YOLO annotations from bounding box coordinates
def save_annotations(filename, boxes, img_width, img_height):
    with open(filename, 'w') as file:
        for box in boxes:
            class_id, x1, y1, x2, y2 = box
            # Convert from absolute to normalized coordinates
            cx = (x1 + x2) / 2 / img_width
            cy = (y1 + y2) / 2 / img_height
            bw = (x2 - x1) / img_width
            bh = (y2 - y1) / img_height
            file.write(f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

# Define the new size for all images
new_size = (640, 640)

# Create directories for the mosaic images and annotations
os.makedirs('data/mosaic_images', exist_ok=True)
os.makedirs('data/mosaic_annotations', exist_ok=True)

# List all images and shuffle them to create random mosaics
image_files = os.listdir('data/images')
np.random.shuffle(image_files)

for i in range(0, len(image_files), 4):
    # Load images and annotations
    images = []
    all_boxes = []
    for img_file in image_files[i:i+4]:
        img_path = f'data/images/{img_file}'
        img = Image.open(img_path).resize(new_size)
        images.append(img)
        ann_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        boxes = load_annotations(f'data/labels/{ann_file}', *img.size)
        all_boxes.append(boxes)

    # Create the mosaic image
    mosaic_image = Image.new('RGB', (new_size[0] * 2, new_size[1] * 2))

    # Initialize list to hold updated annotations
    updated_boxes = []

    # Place images into the mosaic
    for j, img in enumerate(images):
        x_offset = (j % 2) * new_size[0]
        y_offset = (j // 2) * new_size[1]
        mosaic_image.paste(img, (x_offset, y_offset))

        # Adjust annotations for the new image positions and sizes
        for box in all_boxes[j]:
            class_id, x1, y1, x2, y2 = box
            # Adjust for the resized image
            x1 = x1 * new_size[0] / img.size[0]
            y1 = y1 * new_size[1] / img.size[1]
            x2 = x2 * new_size[0] / img.size[0]
            y2 = y2 * new_size[1] / img.size[1]
            # Adjust for the position in the mosaic
            x1 += x_offset
            y1 += y_offset
            x2 += x_offset
            y2 += y_offset
            updated_boxes.append((class_id, x1, y1, x2, y2))

    # Save the mosaic image
    mosaic_filename = f'mosaic_images/{image_files[i][:-4]}_mosaic.jpg'
    mosaic_image.save(mosaic_filename)

    # Save the updated annotations
    save_annotations(f'mosaic_annotations/{image_files[i][:-4]}_mosaic.txt', updated_boxes, new_size[0] * 2, new_size[1] * 2)

print("Mosaic creation complete!")

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
        img = Image.open(img_path)
        images.append(img)
        ann_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        boxes = load_annotations(f'data/labels/{ann_file}', *img.size)
        all_boxes.append(boxes)

    # Determine the size of the mosaic image
    total_width = max(img.size[0] for img in images) * 2
    total_height = max(img.size[1] for img in images) * 2
    mosaic_image = Image.new('RGB', (total_width, total_height))

    # Initialize list to hold updated annotations
    updated_boxes = []

    # Place images into the mosaic
    for j, img in enumerate(images):
        x_offset = (j % 2) * max(img.size[0] for img in images)
        y_offset = (j // 2) * max(img.size[1] for img in images)
        mosaic_image.paste(img, (x_offset, y_offset))

        # Adjust annotations for the new image positions
        for box in all_boxes[j]:
            class_id, x1, y1, x2, y2 = box
            x1 += x_offset
            y1 += y_offset
            x2 += x_offset
            y2 += y_offset
            updated_boxes.append((class_id, x1, y1, x2, y2))

    # Save the mosaic image
    mosaic_filename = f'data/mosaic_images/{image_files[i][:-4]}_mosaic.jpg'
    mosaic_image.save(mosaic_filename)

    # Save the updated annotations
    save_annotations(f'data/mosaic_annotations/{image_files[i][:-4]}_mosaic.txt', updated_boxes, total_width, total_height)

print("Mosaic creation complete!")

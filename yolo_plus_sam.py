import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
from PIL import Image

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth" #your path to .pth
model_type = "vit_h" #replace with the model type you are using

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

predictor = SamPredictor(sam)

# Folder containing YOLO annotation text files
yolo_annotations_folder = 'path/to/your/yolo_labels_folder'

# Folder containing images
images_folder = 'path/to/your/image_folder'
output_folder = 'path/where/you/want/to/save/segmented_images'  # Specify the new output folder
# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Process each YOLO annotation file
for yolo_annotation_file in os.listdir(yolo_annotations_folder):
    if yolo_annotation_file.endswith('.txt'):
        yolo_annotation_path = os.path.join(yolo_annotations_folder, yolo_annotation_file)

        # Read each line from the YOLO annotation file
        with open(yolo_annotation_path, 'r') as f:
            lines = f.readlines()

        # Read the corresponding image
        image_file = yolo_annotation_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        boxes = []

        # Process each annotation
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # Skip invalid lines

            class_id, center_x, center_y, box_width, box_height, confifence = map(float, parts)

            # Convert coordinates back to pixel values
            image_height, image_width, _ = image.shape
            x_min = int((center_x - box_width / 2) * image_width)
            y_min = int((center_y - box_height / 2) * image_height)
            x_max = int((center_x + box_width / 2) * image_width)
            y_max = int((center_y + box_height / 2) * image_height)

            # Append the box coordinates to the list
            boxes.append([x_min, y_min, x_max, y_max])

        # Convert the list of boxes into a tensor
        input_boxes = torch.tensor(boxes, device=predictor.device)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Set up the figure without displaying it
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        #plotting masks
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # #plotting boxes
        # for box in input_boxes:
        #     show_box(box.cpu().numpy(), plt.gca())
        plt.axis('off')

        # Generate the new image file name
        output_image_file = os.path.join(output_folder, image_file.replace('.jpg', '_segmented.jpg'))

        # Save the figure to a temporary buffer
        plt.savefig('temp_output.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to prevent it from displaying

        # Open the temporary image and resize it
        temp_image = Image.open('temp_output.jpg')
        resized_image = temp_image.resize((644, 644), Image.Resampling.LANCZOS)

        # Save the resized image with the new file name
        resized_image.save(output_image_file)

        # Clean up the temporary file
        os.remove('temp_output.jpg')

print("All images segmented successfully!")


# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# for mask in masks:
#     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
# for box in input_boxes:
#     show_box(box.cpu().numpy(), plt.gca())
# plt.axis('off')
# plt.savefig('output.jpg')
# plt.show()

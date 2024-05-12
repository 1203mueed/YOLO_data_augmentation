import os
import cv2

# Paths
flipped_images_folder = 'horizontally_flipped/images'
annotations_folder = 'data/labels'
flipped_annotations_folder = 'horizontally_flipped/labels'
os.makedirs(flipped_annotations_folder, exist_ok=True)

# Process each flipped image
for image_file in os.listdir(flipped_images_folder):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(flipped_images_folder, image_file)
        annotation_file = os.path.join(annotations_folder, image_file.replace('_horiz.jpg', '.txt'))
        flipped_annotation_file = os.path.join(flipped_annotations_folder, image_file.replace('.jpg', '.txt'))

        # Read the flipped image
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Read the original annotation
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        flipped_lines = []
        for line in lines:
            class_id, center_x, center_y, box_width, box_height = map(float, line.strip().split())
            flipped_x = 1.0 - center_x  # Horizontal flip

            # Append the updated annotation
            flipped_lines.append(f"{int(class_id)} {flipped_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

        # Save the flipped annotation
        with open(flipped_annotation_file, 'w') as f:
            f.writelines(flipped_lines)

        print(f"Flipped annotation saved as '{flipped_annotation_file}'")

print("All annotations updated and saved successfully!")

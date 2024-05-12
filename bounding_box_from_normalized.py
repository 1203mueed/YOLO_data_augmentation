import os
import cv2

# Folder containing YOLO annotation text files
yolo_annotations_folder = 'cropped_annotations'

# Folder containing images
images_folder = 'cropped_images'
output_folder = 'cropped_bounding_box'  # Specify the new output folder

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define class names
class_names = {
    0: 'litter',
    1: 'pile',
    2: 'face mask',
    3: 'trash bin',
    4: 'plastic bag',
    5: 'bottle',
    6: 'cup',
    7: 'rope',
    8: 'sachet',
    9: 'straw'
}

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

        # Process each annotation
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip invalid lines

            class_id, center_x, center_y, box_width, box_height = map(float, parts)

            # Convert coordinates back to pixel values
            image_height, image_width, _ = image.shape
            x_min = int((center_x - box_width / 2) * image_width)
            y_min = int((center_y - box_height / 2) * image_height)
            x_max = int((center_x + box_width / 2) * image_width)
            y_max = int((center_y + box_height / 2) * image_height)

            # Draw bounding box
            color = (0, 255, 0)  # Green color
            thickness = 2
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

            # Add class name with increased font size
            class_name = class_names.get(int(class_id), 'Unknown')
            color2 = (255, 2, 2)
            font_scale = 1  # Adjust the font size as needed
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color2, 3)

        # Save the annotated image in the new output folder
        output_image_file = os.path.join(output_folder, image_file.replace('.jpg', '_annotated.jpg'))
        cv2.imwrite(output_image_file, image)

        print(f"Annotated image saved as '{output_image_file}'")

print("All images annotated successfully!")

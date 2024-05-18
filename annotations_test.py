import os
import cv2

# Folder containing annotation text files (COCO format)
annotations_folder = 'path/to/your/yolo_annotation_folder'

# Folder containing images
images_folder = 'path/to/your/image_folder'

# Output folder for annotated images
output_folder = 'path/to/your/output_folder'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each annotation file
for annotation_file in os.listdir(annotations_folder):
    if annotation_file.endswith('.txt'):
        annotation_path = os.path.join(annotations_folder, annotation_file)

        # Read each line from the annotation file
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        # Read the corresponding image
        image_file = annotation_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_folder, image_file)

        # Check if OpenCV can read the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_path}. Skipping...")
                continue
        except Exception as e:
            print(f"Error reading image: {image_path}. Error details: {str(e)}")
            continue

        # Process each annotation
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip invalid lines

            class_name, x_min, y_min, width, height = parts

            # Convert coordinates to integers
            x_min, y_min, width, height = map(float, [x_min, y_min, width, height])
            x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)
            x_max, y_max = x_min + width, y_min + height

            # Draw bounding box
            color = (0, 165, 255)  # Orangy color (BGR format)
            thickness = 20
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

            # Add class label above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (255, 255, 255)  # White color for the label
            cv2.putText(image, class_name, (x_min, y_min - 10), font, font_scale, font_color, thickness)

        # Save the annotated image
        output_image_file = os.path.join(output_folder, image_file.replace('.jpg', '_annotated.jpg'))
        cv2.imwrite(output_image_file, image)

        print(f"Annotated image saved as '{output_image_file}'")

print("All images annotated successfully!")

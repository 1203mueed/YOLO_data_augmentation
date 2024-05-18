# YOLO_data_augmentation

This repo contains .py files to augment images to use in YOLO format. You can also use this repo to augment other models. Then, you have to adjust the annotations like this:

    #Process each annotation
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
            
*** For all code, remember to edit paths
*** For all types of augmentations, new images and annotations are created

*** If you want to randomly flip 25% of your whole dataset horizontally, then run horizontal_flip.py(this code will flip images horizontally) and horizontal_flip_annotations.py(this will create annotations). 

*** If you want to change the brightness and contrast of the 25% of your dataset, run the brightness_contrast.py. You can change the brightness and contrast values.

*** If you run the cropped_images.py, it will randomly select 25% of the images, and it will crop the bounding boxes those area is less than 15% of the total image. The difference between small_images.py and cropped_images.py is that cropped_images.py resized the images into 644/644. And  small_images.py keeps the original size.

*** updated_mosaic.py randomly selects four images and combines them.

*** If you want to check your annotations, you can use bounding_box_normalized.py. This code will draw the bounding boxes from the annotations in the image.

### YOLO & SAM

*** To segment the bounding box area using SAM(segment anything model), you can run yolo_plus_sam.py.

#### Setup for yolo_plus_sam.py
Follow the GitHub repo of Segment Anything Model : https://github.com/facebookresearch/segment-anything

Download a model as instructed in the SAM repo. Update the model path and the model type in the yolo_plus_sam.py file.

*** If you want to plot the boxes along with the masks, comment out these lines:

        ##plotting boxes
        # for box in input_boxes:
        #     show_box(box.cpu().numpy(), plt.gca())


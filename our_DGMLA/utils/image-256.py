import os
from PIL import Image


def resize_images_in_directory(directory, output_directory=None):
    # Use input directory as default if no output specified
    if output_directory is None:
        output_directory = directory

    # Iterate through all files in directory
    for filename in os.listdir(directory):
        # Check for image file extensions
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            input_path = os.path.join(directory, filename)
            # Open and resize image to 256x256
            with Image.open(input_path) as img:
                resized_img = img.resize((256, 256))
                # Save resized image
                output_path = os.path.join(output_directory, filename)
                resized_img.save(output_path)


# Specify directories
input_directory = 'datatest/UFO-120/TEST/testA'
# Optional: Specify output directory
output_directory = 'datatest/UFO-120/TEST/test'

# Run resizing function
resize_images_in_directory(input_directory, output_directory)
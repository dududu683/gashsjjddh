import os
from PIL import Image


def resize_images_in_directory(directory, output_directory=None):
    """Resize all images in directory to 256x256.

    Args:
        directory (str): Path to input directory containing images
        output_directory (str, optional): Path to save resized images.
            Uses input directory if not specified.
    """
    # Use input directory as output if not specified
    if output_directory is None:
        output_directory = directory

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all files in directory
    for filename in os.listdir(directory):
        # Check for image file extensions
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Open, resize and save image
            with Image.open(input_path) as img:
                resized_img = img.resize((256, 256))
                resized_img.save(output_path)
                print(f"Resized: {filename} -> Saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Configuration
    INPUT_DIR = 'test/reference_90'
    OUTPUT_DIR = 'testdata/target/'

    # Run resizing
    resize_images_in_directory(INPUT_DIR, OUTPUT_DIR)
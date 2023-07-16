import os
import re
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm
import openai

root = tk.Tk()
root.withdraw()

def select_folder(prompt_text):
    print(prompt_text)
    folder_path = filedialog.askdirectory()
    print(f"Selected Directory: {folder_path}")
    return folder_path

def input_transformation_value(prompt_text):
    print(prompt_text)
    value = simpledialog.askstring("Input", prompt_text)
    return float(value) if value else None

def yes_no_prompt(prompt_text):
    print(prompt_text)
    return simpledialog.askstring("Input", prompt_text).lower() == 'yes'

# Select input/output folders
input_folder = select_folder("Please select the input directory of images.")
output_folder = select_folder("Please select the output directory for augmented images.")

# Prompt for GPT guidance
use_gpt = yes_no_prompt("Do you want to use ChatGPT for transformation recommendations? (yes/no): ")

transformations = {}

if use_gpt:
    # Set OpenAI key
    openai.api_key = simpledialog.askstring("Input", "Enter your OpenAI API Key:", show='*')

    # Get image description from the user
    image_desc = simpledialog.askstring("Input", "Please describe the type of images you have: ")

    # Request augmentation suggestions from ChatGPT
    print("Requesting augmentation suggestions from ChatGPT...")
    chat_model = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that understands image augmentation for training deep learning models, and how to best augment images depending on what type of image is being augmented."},
            {"role": "user", "content": f"I have pictures of {image_desc}. How should I augment them for a machine learning model? Please provide your recommendations as a list of transformations with each transformation followed by a numeric value or range. For example, 'flip_lr: 0.5, rotate: -20 to 20, brightness: 0.8 to 1.2'. Please include transformations for flip_lr, rotate, brightness, contrast, zoom, noise, shear, and grayscale. Please do not include explanatory text with the recommendations."}
        ])

    gpt_response = chat_model['choices'][0]['message']['content']

    # Parse the response
    gpt_recommendations = gpt_response.split('\n')
    for recommendation in gpt_recommendations:
        if ':' in recommendation:  # Only consider lines that contain ':' (transformation:value pairs)
            # Remove numerical indexes before transformations
            recommendation = re.sub(r"^\d+\.\s*", "", recommendation)

            transformation, value = recommendation.split(':', 1)  # Split only once
            transformation = transformation.strip().lower()

            # Clean up the value, removing any comments after '#'
            value = value.split('#')[0].strip()

            # Handle different transformation types
            try:
                if "to" in value:  # Range
                    value = tuple(map(float, value.split("to")))
                elif "True" in value or "False" in value:  # Boolean
                    value = value.strip() == "True"
                else:  # Single value
                    value = float(value)

                transformations[transformation] = value
                print(f"Transformation recommended: {transformation} Value recommended: {value}")

            except ValueError as e:
                print(f"Error while parsing value for transformation '{transformation}': {e}")
                continue


# If GPT wasn't used, get transformations manually
else:
    transformations = {
        'flip_lr': input_transformation_value("Enter the flip_lr value (0.0 - 1.0): Flip Left/Right. A high value (near 1) means high likelihood of flipping images left/right.") or 0.5,
        'rotate': (input_transformation_value("Enter the minimum rotate value (-45 - 45): Rotate the images by a certain degree. Negative values mean rotating clockwise, positive values mean rotating counterclockwise.") or -25, input_transformation_value("Enter the maximum rotate value (-45 - 45): ") or 25),
        'brightness': (input_transformation_value("Enter the minimum brightness value (0.0 - 3.0): Adjust the brightness of images. Values less than 1 mean darker images, values greater than 1 mean brighter images.") or 0.8, input_transformation_value("Enter the maximum brightness value (0.0 - 3.0): ") or 1.2),
        'contrast': (input_transformation_value("Enter the minimum contrast value (0.0 - 2.0): Adjust the contrast of images. Values less than 1 mean less contrast, values greater than 1 mean more contrast.") or 0.8, input_transformation_value("Enter the maximum contrast value (0.0 - 2.0): ") or 1.2),
        'zoom': (input_transformation_value("Enter the minimum zoom value (0.5 - 1.5): Zoom into images. Values less than 1 mean zooming in, values greater than 1 mean zooming out.") or 0.8, input_transformation_value("Enter the maximum zoom value (0.5 - 1.5): ") or 1.2),
        'noise': (input_transformation_value("Enter the minimum noise value (0.0 - 0.2): Add gaussian noise to images. A higher value means more noise.") or 0.0, input_transformation_value("Enter the maximum noise value (0.0 - 0.2): ") or 0.05),
        'shear': (input_transformation_value("Enter the minimum shear value (-25 - 25): Shear (tilt) images. Negative values mean shearing to the left, positive values mean shearing to the right.") or -10, input_transformation_value("Enter the maximum shear value (-25 - 25): ") or 10),
        'grayscale': input_transformation_value("Enter the grayscale value (0.0 - 1.0): Convert images to grayscale. A value of 0 means no conversion to grayscale, a value of 1 means full conversion to grayscale.") or 0
    }

# Define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(transformations.get('flip_lr', 0.5)),
    iaa.Affine(rotate=transformations.get('rotate', (-25, 25))),
    iaa.MultiplyBrightness(transformations.get('brightness', (0.8, 1.2))),
    iaa.LinearContrast(transformations.get('contrast', (0.8, 1.2))),
    iaa.Affine(scale=transformations.get('zoom', (0.8, 1.2))),
    iaa.AdditiveGaussianNoise(scale=transformations.get('noise', (0.0, 0.05))),
    iaa.Affine(shear=transformations.get('shear', (-10, 10))),
    iaa.Grayscale(alpha=transformations.get('grayscale', 0))
])

# Iterate over images in the input folder
image_files = os.listdir(input_folder)
progress_bar = tqdm(total=len(image_files), ncols=75, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    try:
        image = np.array(Image.open(input_path).convert("RGB"))  # Convert RGBA images to RGB
    except Exception as e:
        print(f"Could not open image file: {input_path}, skipping. Error: {e}")
        continue

    # Augment the image
    augmented_image = seq(image=image)

    # Save the augmented image
    output_path = os.path.join(output_folder, f"aug_{image_file}")
    Image.fromarray(augmented_image).save(output_path)

    progress_bar.update(1)

progress_bar.close()
print("Augmentation completed!")

import os
import re
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm
import openai

# Initialize the main window
root = tk.Tk()
root.withdraw()

# Function to select a folder
def select_folder(prompt_text):
    print(prompt_text)
    folder_path = filedialog.askdirectory()
    print(f"Selected Directory: {folder_path}")
    return folder_path

# Function to input transformation values
def input_transformation_value(prompt_text):
    print(prompt_text)
    value = simpledialog.askstring("Input", prompt_text)
    return float(value) if value else None

# Function for yes/no prompts
def yes_no_prompt(prompt_text):
    print(prompt_text)
    return simpledialog.askstring("Input", prompt_text).lower() == 'yes'

# Function to manage OpenAI API key
def manage_api_key():
    api_key_file = 'openai_api_key.txt'
    if os.path.exists(api_key_file):
        with open(api_key_file, 'r') as file:
            return file.read().strip()
    else:
        api_key = simpledialog.askstring("Input", "Enter your OpenAI API Key:", show='*')
        with open(api_key_file, 'w') as file:
            file.write(api_key)
        return api_key

# Select input/output folders
input_folder = select_folder("Please select the input directory of images.")
output_folder = select_folder("Please select the output directory for augmented images.")

# Prompt for GPT guidance
use_gpt = yes_no_prompt("Do you want to use ChatGPT for transformation recommendations? (yes/no): ")

transformations = {}

if use_gpt:
    openai.api_key = manage_api_key()
    
    # Get image description from the user
    image_desc = simpledialog.askstring("Input", "Please describe the type of images you have: ")

    # Request augmentation suggestions from ChatGPT
    print("Requesting augmentation suggestions from ChatGPT...")
    chat_model = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that understands image augmentation for training deep learning models."},
            {"role": "user", "content": f"I have pictures of {image_desc}. Can you provide augmentation suggestions for a machine learning model? List each transformation followed by a numeric value or range on a new line. Use simple formatting: one transformation per line, name and value separated by a colon, and ranges indicated with 'to'. For example: \nflip_lr: 0.5\nrotate: -20 to 20\nbrightness: 0.8 to 1.2\n... Include transformations for flip_lr, rotate, brightness, contrast, zoom, noise, shear, and grayscale."}
        ]
    )

    gpt_response = chat_model['choices'][0]['message']['content']


    # Parse the response
    gpt_recommendations = gpt_response.split('\n')
    for recommendation in gpt_recommendations:
        if ':' in recommendation:
            transformation, value = recommendation.split(':', 1)
            transformation = transformation.strip().lower()
            value = value.split('#')[0].strip()

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

else:
    # Manual input of transformations
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

# Define individual augmentation functions
def apply_flip_lr(image):
    return iaa.Fliplr(1.0).augment_image(image)

def apply_rotate(image, value):
    return iaa.Affine(rotate=value).augment_image(image)

def apply_brightness(image, value):
    return iaa.MultiplyBrightness(value).augment_image(image)

def apply_contrast(image, value):
    return iaa.LinearContrast(value).augment_image(image)

def apply_zoom(image, value):
    return iaa.Affine(scale=value).augment_image(image)

def apply_noise(image, value):
    return iaa.AdditiveGaussianNoise(scale=value).augment_image(image)

def apply_shear(image, value):
    return iaa.Affine(shear=value).augment_image(image)

def apply_grayscale(image, value):
    return iaa.Grayscale(alpha=value).augment_image(image)

# Iterate over images in the input folder
image_files = os.listdir(input_folder)
progress_bar = tqdm(total=len(image_files) * 8, ncols=75, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

# Allowed image file extensions
allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    # Extract the file extension
    _, file_extension = os.path.splitext(image_file)

    # Check if the file extension is in the list of allowed extensions
    if file_extension.lower() not in allowed_extensions:
        print(f"Skipping non-image file: {input_path}")
        continue

    try:
        image = np.array(Image.open(input_path).convert("RGB"))  # Convert image to RGB
    except Exception as e:
        print(f"Error processing image file: {input_path}, skipping. Error: {e}")
        continue

    # Apply each transformation and save
    transformations_sequence = [
        ('flip_lr', apply_flip_lr),
        ('rotate', lambda img: apply_rotate(img, np.random.uniform(*transformations.get('rotate', (-25, 25))))),
        ('brightness', lambda img: apply_brightness(img, np.random.uniform(*transformations.get('brightness', (0.8, 1.2))))),
        ('contrast', lambda img: apply_contrast(img, np.random.uniform(*transformations.get('contrast', (0.8, 1.2))))),
        ('zoom', lambda img: apply_zoom(img, np.random.uniform(*transformations.get('zoom', (0.8, 1.2))))),
        ('noise', lambda img: apply_noise(img, np.random.uniform(*transformations.get('noise', (0.0, 0.05))))),
        ('shear', lambda img: apply_shear(img, np.random.uniform(*transformations.get('shear', (-10, 10))))),
        ('grayscale', lambda img: apply_grayscale(img, transformations.get('grayscale', 0)))
    ]

    for transformation_name, transformation_function in transformations_sequence:
        transformed_image = transformation_function(image)
        output_path = os.path.join(output_folder, f"aug_{transformation_name}_{image_file}")
        Image.fromarray(transformed_image).save(output_path)
        progress_bar.update(1)

progress_bar.close()
print("Augmentation completed!")

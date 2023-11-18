# Image Augmentation Tool

This tool provides a simple way to perform image augmentations using an interactive command-line interface. Image augmentation is a technique used in training AI models, particularly in computer vision tasks, to artificially increase the diversity and quantity of training data. It involves applying various transformations and modifications to the existing images to create new, slightly modified versions.  It uses both manual user input and the GPT-3.5-turbo model from OpenAI for augmentation suggestions based on the type of images you are working with.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have a Windows machine. This tool was developed and tested on Windows 10, but it should work on other versions as well.
- You have Python 3.10 or later installed on your machine.
- You have installed the required Python dependencies. You can install these by running `py -m pip install -r requirements.txt` from your command line.

## Usage

1. Download or clone this repository to your local machine.
2. Open a command line interface (such as Command Prompt or PowerShell) and navigate to the directory containing `ImageAugmentationTool_V2.py`.
3. Run the script by typing `py ImageAugmentationTool_V2.py` and pressing enter.
4. Follow the prompts on the terminal to specify the input and output directories, and to choose the augmentation methods.

If you say "yes" to using chatGPT, it will prompt for an OpenAI key. This key is saved in the local directly from which this application is ran. Next time it is ran, it will first check for any saved keys.

In the manual mode of the augmentation, you will be prompted to enter values for each type of transformation. If you don't want to manually specify a value, you can simply press "enter", and the program will use the following default values:

- flip_lr: 0.5
- rotate: (-25, 25)
- brightness: (0.8, 1.2)
- contrast: (0.8, 1.2)
- zoom: (0.8, 1.2)
- noise: (0.0, 0.05)
- shear: (-10, 10)
- grayscale: 0

**NOTE:** This tool is designed to work with .jpg, .png, .jpeg and .bmp files.

## License

This project uses the following license: MIT License.

This software includes python components that are distributed under the terms of their respective licenses. The Python software is licensed under the Python Software Foundation License (PSFL).

You may need to install additional software to use this tool, which may be governed by additional licenses.

The chat feature of this tool uses the OpenAI GPT-3.5-turbo model, which is subject to OpenAI's use case policy.

## Contact

Please file issues on the GitHub repository for any problems or feature requests.

# Brain Tumor MRI Image Generation using GANs

This project utilizes Generative Adversarial Networks (GANs) to generate synthetic Brain Tumor MRI images for data augmentation purposes.
Please refer to Project-Report.pdf for better project understanding and overview

## Table of Contents

- [Project Description](#project-description)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Testing](#testing)

## Project Description

The goal of this project is to generate synthetic Brain Tumor MRI images using GANs. Data augmentation is crucial in deep learning, especially in medical imaging where data is often limited. By generating synthetic images, we can increase the size of our dataset and improve the performance of our models.

## Dependencies

The following libraries are required to run this project:

- Numpy
- Tensorflow
- Matplotlib
- OpenCV
- Keras
- Seaborn
- tqdm

You can install them using pip:
bash pip install numpy tensorflow matplotlib opencv-python keras seaborn tqdm

## Dataset

The dataset used in this project is the "Brain Tumor Detection" dataset, which can be found on Kaggle. The dataset contains MRI images of brains with and without tumors. For this project, we are only using the images with tumors (located in the "yes" folder).

## Usage

1. Clone the repository:
2. Download the dataset and place it in the project directory.
3. Run the `main.py` script to train the GAN and generate images.

## Model Architecture

The GAN consists of two main components:

- **Generator:** A neural network that takes random noise as input and generates synthetic images.
- **Discriminator:** A neural network that classifies images as real or fake.

The generator and discriminator are trained against each other in an adversarial manner. The generator tries to generate images that fool the discriminator, while the discriminator tries to correctly classify real and fake images.

## Training

The GAN is trained using the Adam optimizer. The training process involves iteratively updating the weights of the generator and discriminator networks. The training parameters are defined in the `main.py` script.

## Results

The generated images are saved in the `generated_images` directory. The quality of the generated images can be evaluated visually and by comparing their distributions to the real images.

## Testing

The generated images are tested by plotting their distributions against the real images. If the distributions overlap, it indicates that the generated samples are very close to the real ones.

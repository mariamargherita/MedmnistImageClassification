
# MEDMNIST Image Classification

This GitHub repository showcases a project that focuses on classifying medical images obtained from the [MEDMNIST dataset](https://github.com/MedMNIST/MedMNIST).
*MEDMNIST* is a large-scale collection of standardized biomedical images, including 12 datasets for 2D and 6 datasets for 3D. 
All images are pre-processed into 28x28 / 64x64 / 128x128 / 224x224 for 2D and 28x28x28 / 64x64x64 for 3D with the corresponding classification labels. 

For our exploratory use case we will leverage just 2 of the 12 datasets available: *PathMNIST* and *BreastMNIST*.
- The *PathMNIST* is composed of a source dataset of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset of 7,180 image patches from a different clinical center. 
The dataset includes 9 types of tissues, resulting in a multi-class classification task. The source dataset is split into training and validation set with a ratio of 9:1.
- The *BreastMNIST* is a source dataset of 780 breast ultrasound images. It is categorized into 3 classes: normal, benign, and malignant. 
The task haas then been simplified into binary classification by combining normal and benign as positive and classifying them against malignant as negative. 
The source dataset was split with a ratio of 7:1:2 into training, validation and test set.

For more information about *MEDMNIST* please read [this article](https://www.nature.com/articles/s41597-022-01721-8).

> **Note**: This project was carried out in PyCharm, thus it is optimized for it. However, this should not keep you from using your own preferred server.
<br>

<br>

## Project Objective

The primary goal of this project is to address the following problem:

> **Problem**: Our goal is to construct a classifier that can determine the class of a new, unseen example as accurately as possible for both *PathMNIST* and *BreastMNIST* datasets.

The main *metric* that will be used to assess the performance of the models is *accuracy*.
<br>

## Usage

To use this project, follow these steps:

1. **Clone the repository**: First, clone this repository to your local machine using

    ```bash
    git clone https://github.com/mariamargherita/Medmnist_Classification.git
    ```

2. **Obtain the dataset**: To obtain the dataset please run the $data_import.py$ file available in the data folder. Follow the directions in the file to successfully download the data.

3. **Install virtual environment**: This project requires the installation of a specific Conda environment. You can install it by typing the following in your terminal:

    ```bash
    conda env create -f medmnist_classification.yml
    ```
   
You should now be able to run the Notebooks.

<br>

## Project Outline

This repository contains the following files:

   ```
    ├──── ...
    ├──── ...
    └──── utils.py: Python file containing useful functions to run code
   ```

### Data Preprocessing


<br>

### Model Selection and Training

<br>

## Results

<br>

## Contributions


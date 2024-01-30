
# MEDMNIST Image Classification

This GitHub repository showcases a project that focuses on classifying medical images obtained from the [*MEDMNIST dataset*](https://github.com/MedMNIST/MedMNIST).
*MEDMNIST* is a large-scale collection of standardized biomedical images, including 12 datasets for 2D and 6 datasets for 3D. 
All images are pre-processed into 28x28 / 64x64 / 128x128 / 224x224 for 2D and 28x28x28 / 64x64x64 for 3D with the corresponding classification labels. 

For our exploratory use case we will leverage just 2 of the 12 datasets available: *PathMNIST* and *BreastMNIST*.
- The *PathMNIST* is composed of a source dataset of 100,000 non-overlapping RGB image patches from hematoxylin & eosin stained histological images, and a test dataset of 7,180 RGB image patches from a different clinical center. 
The dataset includes 9 types of tissues, resulting in a multi-class classification task. The source dataset is split into training and validation set with a ratio of 9:1.
- The *BreastMNIST* is a source dataset of grey-scale 780 breast ultrasound images. It is categorized into 3 classes: normal, benign, and malignant. 
The task haas then been simplified into binary classification by combining normal and benign as positive and classifying them against malignant as negative. 
The source dataset was split with a ratio of 7:1:2 into training, validation and test set.

For more information about *MEDMNIST* please read [this article](https://www.nature.com/articles/s41597-022-01721-8).

> **Note**: This project was carried out in PyCharm, thus it is optimized for it. However, this should not keep you from using your own preferred server.
<br>

<br>

## Project Objective

The primary goal of this project is to address the following problem:

> **Problem**: Our goal is to explore classifiers that can determine the class of a new, unseen example as accurately as possible for both *PathMNIST* and *BreastMNIST* datasets.
> **Disclaimer**: This will be more an exploration of the *MEDMNIST* datasets, models and techniques than a results oriented analysis, thus our overall performance might result unsatisfactory for users who look for great results.

The main *metric* that will be used to assess the performance of the models is *accuracy*.
<br>

## Usage

To use this project, follow these steps:

1. **Clone the repository**: First, clone this repository to your local machine using

    ```bash
    git clone https://github.com/mariamargherita/Medmnist_Classification.git
    ```

2. **Obtain the dataset**: To obtain the dataset please run the *data_import.py* file available in the data folder. Follow the directions in the file to successfully download the data.

3. **Install virtual environment**: This project requires the installation of a specific Conda environment. You can install it by typing the following in your terminal:

    ```bash
    conda env create -f medmnist_classification_env.yml
    ```
   
You should now be able to run the Notebooks.

<br>

## Project Outline

This repository contains the following files:

   ```
    ├──── data foder: contains the data import.py file to import data
    ├──── models folder: contains the checkpoint of best CNN model with respect to validation accuracy performance
    ├──── plots foder: contains PCA visualizations, accuracy vs. validation accuracy and loss vs. validation loss
    ├──── breast_pipeline.py: Python file containing the project pipeline for BreastMNIST data classification
    ├──── cnn_model.py: Python file containing the CNN model
    ├──── medmnist_classification_env.yml: .yml file containing the Conda environment needed to run the code
    ├──── path_pipeline.py: Python file containing the project pipeline for PathMNIST data classification
    └──── utils.py: Python file containing useful functions to run code
    
   ```

### Data Load and Preprocessing

Since the data owners already provided quite clean datasets and the training set, validation set and test set were already imported by running the *data_import.py* file, the data loading phase just requires to use the *data_feed()* function available in the *utils.py* file.
Indeed, this is the first step performed in the pipelines.

Preprocessing steps performed on *PathMNIST*:
- pixels normalization by dividing for 255

>**Note**: the conversion step for images from RGB to grey scale was not performed since the model performed best on RGB images
but the code was kept in the pipeline for reference.

Preprocessing steps performed on *BreastMNIST*:
- images were reshaped and scaled with mean 0 and standard deviation 1 to apply PCA
- the training data were augmented with horizontal/vertical flips and random rotations

>**Note**: the conversion step from RGB to grey-scale was not necessary in the *BreastMNIST* dataset since
the images are already provided in grey-scale.

<br>

### Model Selection and Training

- *PathMNIST* dataset
    
    We tried different model complexities and tuned parameters and hyperparameters. We also tried different batch sizes since this could have a strong impact on how well the neural network learns to generalize. Finally, we made sure to add dropout and early stopping to limit over fitting.

- *BreastMNIST* dataset
    
    We did cross validation grid search to tune the random forest hyperparameters.

<br>

## Results

- *PathMNIST* dataset 
    
    For exploration purposes, we fit the model on both RGB images and grey-scale images. We then decided to keep the RGB model since it held better performance results on the test set.
    
    We trained the model on 90% of training data and reserved a 10% for validation data. Once we found the model with the best performance on the validation data, we made predictions on the test date getting a *test accuracy of 85%*.
      
    > For exploration purposes, in the pipeline we left the code for training the best model on the full training data with prediction of test data labels and respective print of performance results.


- *BreastMNIST* dataset 

    We tuned the random forest hyperparameters on all training data by using cross validation. Once we found the best model, we made predictions on the test data getting a *test accuracy of 73%*.
    
    > **Note**: This test accuracy is quite low and would need to be improved by means of better hyperparameter tuning. However, the computational effort is quite significant following data augmentation.


<br>

## Contributions

Here are some of the steps that could still be taken in order to potentially improve the models:

- *PathMNIST*:
  - Try different model complexities
  - Try RMSprop optimizer on grey-scale image model and re-tune parameters and hyperparameters if necessary
  >**Note**: the RMSprop optimizer was tested on RGB model but the Adam optimizer held better results. However, it was not tested on grey-scale model.
  - Tune parameters for Adam optimizer (i.e. learning rate)
  
- *BreastMNIST*:
  - Perform a more intense grid search for Random Forest
  - Play around with the data augmentation step to see if the model performance can be improved with different augmentation strategies
  - Try different models which could for sure perform better than Random Forest
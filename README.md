# Self-Paced Data Augmentation
## Overview
This repo contains the source code of our project, "Self-Paced Data Augmentation", which studies how to improve the generalization ability of models to unseen domains by augmenting and presenting examples in a meaningful order.
We design a self-paced training scheduler that organizes example augmentation and presentation based on their difficulty, and introduce a gradient confidence difficulty measure that captures the magnitude and uncertainty of decision boundary changes caused by the original example and its augmented version.

# Get Started
## 1. Datasets
Our framework currently support five popular benchmark datasets:
* **Digits**: Digits consists of four digit recognition tasks, namely MNIST, MNIST-M, SVHN, and SYN. MNIST contains handwritten digit images, while MNIST-M blends the images in MNIST with random color patches. SVHN contains images of street view house numbers, and SYN contains synthetic digit images with different fonts, backgrounds, and stroke colours. The domain shift in Digits corresponds to font style and background change.
* **PACS**: 
* Office_Home
* VLCS
* NICO++

**Note**: Digits, PACS, Office_Home, and VLCS datasets are small. They will be automatically downloaded when you run the codes. 


## 2. Models
Our framework currently support four base augmentation methods: 
* ERM
* CrossGrad
* DomainMix
* DDAIG

## 3. Difficulty Measures
Our framework currently support four difficulty measures:
* GCDM
* Loss
* Confidence
* Gradients

## 4. Training




Please follow the following formate to run experiments:
                python train.py 
                
                --gpu 0                                                           # GPU 

                --seed 42                                                         # Random Seed
                
                --dataset_path datasets --output_dir output/ERM_DIGITS_MNIST      # Dataset Directory 
                
                --trainer ERM                                                     # Base Augmentation Methods 
                
                --curriculum GCDM                                                 # Difficulty Measure 
                
                --eta 1.00                                                        # Pace Parameter
                
                --source_domains mnist_m svhn syn                                 # Source Domains
                
                --target_domain mnist                                             # Target Domain
                
                --config_path_trainer configs/trainers/ERM/digits.yaml            # Config File for Base Augmentation Method
                
                --config_path_dataset configs/datasets/digits.yaml                # Config File for Datasets

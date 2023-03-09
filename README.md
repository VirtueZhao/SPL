# Self-Paced Data Augmentation
## Overview
This repo contains the source code of our project, "Self-Paced Data Augmentation", which studies how to improve the generalization ability of models to unseen domains by augmenting and presenting examples in a meaningful order.
We design a self-paced training scheduler that organizes example augmentation and presentation based on their difficulty, and introduce a gradient confidence difficulty measure that captures the magnitude and uncertainty of decision boundary changes caused by the original example and its augmented version.

# Get Started
## 1. Datasets
Our framework currently support five popular benchmark datasets:
* **Digits**: Digits consists of four digit recognition tasks, namely MNIST, MNIST-M, SVHN, and SYN. MNIST contains handwritten digit images, while MNIST-M blends the images in MNIST with random color patches. SVHN contains images of street view house numbers, and SYN contains synthetic digit images with different fonts, backgrounds, and stroke colours. The domain shift in Digits corresponds to font style and background change.

* **PACS**: PACS consists of four domains, namely Photo (1,670 images), Art Painting (2,048 images), Cartoon (2,344 images) and Sketch (3,929 images). Each domain contains seven categories. The domain shift in PACS corresponds to image style changes.

* **Office_Home**: Office-Home is initially introduced for domain adaptation and getting popular in the DG community. It contains four domains: Artistic, Clipart, Product, and Real World, where each domain has 65 classes related to office and home objects. There are 15,500 images in total, with an average of around 70 images per class and a maximum of 99 images in each class. The domain shift corresponds to the background, viewpoint and image style changes.
 
* **VLCS**: VLCS consists of four domains of data collected from Caltech101, PASCAL, LabelMe, and SUN, where five common categories are collected: bird, car, chair, dog and person. The domain shift corresponds to background and viewpoint changes.

* **NICO++**: NICO++ is the latest DG dataset that was constructed in 2022 for OOD (Out-of-Distribution) image classification. The public version contains six domains: Autumn, Dim, Grass, Outdoor, Rock, and Water, where each domain has 60 classes. The domain shift corresponds to the context changes. Compared with the previous four datasets, NICO++ is much larger in scale with 88,866 images in total.

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

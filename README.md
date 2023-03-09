# Self-Paced Data Augmentation
## Overview
This repo contains the source code of our project, "Self-Paced Data Augmentation", which studies how to improve the generalization ability of models to unseen domains by augmenting and presenting examples in a meaningful order.
We design a self-paced training scheduler that organizes example augmentation and presentation based on their difficulty, and introduce a gradient confidence difficulty measure that captures the magnitude and uncertainty of decision boundary changes caused by the original example and its augmented version.

# Get Started
## 1. Datasets
## 2. Models
## 3. Difficulty Measures
## 4. Training

Current Available Base Augmentation Method: [ERM, CrossGrad, DomainMix, DDAIG]

Current Available Difficulty Measure: [GCDM, Loss, Confidence, Gradients]

Current Available Datasets: [Digits, PACS, Office_Home, VLCS, NICO++]


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

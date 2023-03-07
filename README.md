# SPL

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

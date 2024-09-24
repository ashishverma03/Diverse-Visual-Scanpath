# Scanpath Clustering and HMM-based Generative Data Augmentation
This is MATLAB implementation of scanpath clustering and HMM-based generative data augmentation technique used in our paper "Generative Augmentation Driven Prediction of Diverse Visual Scanpaths in Images".

## Folder Structure to be used in the Matlab code
```bash
parentfolder
    └── Resized_Traj_image
            ├── name1.mat
            ├── .....
            └── nameN.mat
    └── HMM_gen_traj_label
            ├── name1.mat
            ├── .....
            └── nameN.mat
    └── Labels_org_traj
            ├── name1.mat
            ├── name2.mat
            ├── .....
            └── nameN.mat
```

* The 'Resized_Traj_image' folder contains scanpaths of images of a dataset. For each image, a '.mat' file contains a cell array of dimension 1xN which contains N scanpaths of size 2xL, where L is the scanpath length.
* The 'HMM_gen_traj_label' folder contains scanpaths generated from image-specific HMMs. For each image, a '.mat' file contains a cell array of dimension 2xM which contains M HMM-generated scanpaths and corresponding labels (agreement score).
* The 'Labels_org_traj' folder contains the labels (agreement scores) for human scanpaths. For each image, a '.mat' file contains an array of dimension 1xN which contains the agreement scores for each scanpaths of an image in the input folder 'Resized_Traj_image'.

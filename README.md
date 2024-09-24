# Generative Augmentation Driven Prediction of Diverse Visual Scanpaths in Images
This repository contains the PyTorch implementation for our IEEE TAI paper:

**Ashish Verma, Debashis Sen. Generative Augmentation Driven Prediction of Diverse Visual Scanpaths in Images. [Webpage](https://ashishverma03.github.io/Diverse-Visual-Scanpath)**

## Requirements
* Python 3.7
* PyTorch 0.4.1 (advanced version can also be used with slight modifications)
* We also provide a requirement.txt file.

## Test Codes

* Store your test images inside the 'test' folder in the [dataset](https://github.com/ashishverma03/Diverse-Visual-Scanpath/tree/main/dataset) folder in the root directory. 
* To run the code, you need to download the pre-trained model from [here](https://drive.google.com/drive/u/1/folders/18MQwqiqVuIn5sGf2ngZlRj2_PZSeXFWN), and extract files to the [models](https://github.com/ashishverma03/Diverse-Visual-Scanpath/tree/main/models) folder in the root directory.
* Set argument --mode to 0 for generating scanpaths of different varieties and 1 for generating multiple scanpaths of the same variety.

```run the code
python test.py --mode 0
```
* Check [Results](https://github.com/ashishverma03/Diverse-Visual-Scanpath/tree/main/Results) folder for generated scanpaths of either mode 0 or 1.

## Train Codes

* The code for scanpath clustering and HMM-based generative augmentation is provided in the [Preporcessing](https://github.com/ashishverma03/Diverse-Visual-Scanpath/tree/main/Preprocessing) folder. 
* The pre-processed OSIE dataset used in training is provided [here](https://drive.google.com/drive/u/1/folders/18MQwqiqVuIn5sGf2ngZlRj2_PZSeXFWN). The images are resized to 512x512 and the corresponding scanpaths are scaled down to match the image size. Download it and keep extracted train, val and test folders in the [dataset](https://github.com/ashishverma03/Diverse-Visual-Scanpath/tree/main/dataset) folder in the root directory.
* Check arguments in the [train.py](https://github.com/ashishverma03/Diverse-Visual-Scanpath/blob/main/train.py), for different configurations and settings.

  ``` run the codes
  python train.py
  ```
* Check the [models](https://github.com/ashishverma03/Diverse-Visual-Scanpath/tree/main/models) folder in the root directory for trained models.

## Reference
Please consider citing the paper if you find the code useful in your research.
```
@ARTICLE{Verma2024generative,
  author={Verma, Ashish and Sen, Debashis},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={Generative Augmentation-Driven Prediction of Diverse Visual Scanpaths in Images}, 
  year={2024},
  volume={5},
  number={2},
  pages={940-955},
  keywords={Visualization;Hidden Markov models;Predictive models;Computational modeling;Training;Task analysis;Deep learning;Diverse visual scanpath prediction;generative data augmentation;long short-term memory (LSTM)-based prediction},
  doi={10.1109/TAI.2023.3278650}
}
```



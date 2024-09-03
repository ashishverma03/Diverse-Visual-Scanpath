# Generative Augmentation Driven Prediction of Diverse Visual Scanpaths in Images
This repository contains the PyTorch implementation for our IEEE TAI paper:

**Ashish Verma, Debashis Sen. Generative Augmentation Driven Prediction of Diverse Visual Scanpaths in Images.**

## Requirements
* Python 3.7
* PyTorch 0.4.1 (advanced version can also be used with slight modifications)
* We also provide a requirement.txt file.

## Test Codes

* Store your test images inside the 'test' folder in the dataset folder in the root directory. 
* To run the code, you need to download the pre-trained model from here, and extract files to the models folder in the root directory. 

```run the code
python test.py --mode 0
```

## Reference
If you find the code useful in your research, please consider citing the paper.
```
@ARTICLE{10130633,
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



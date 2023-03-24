# FGSM Attack on Traffic Sign Classifier

This project demonstrates the Fast Gradient Sign Method (FGSM) adversarial attack on a Traffic Sign Classifier trained on a custom dataset. The goal is to showcase the vulnerability of deep learning models to adversarial examples and provide insights into potential defense mechanisms.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Traffic Sign Classifier](#training-the-traffic-sign-classifier)
  - [Performing the FGSM Attack](#performing-the-fgsm-attack)
- [License](#license)

## Overview

The project contains two main Python scripts:

1. `train.py`: Trains a deep learning model (TrafficSignNet) to detect various road signs.
2. `fgsm.py`: Performs the Fast Gradient Sign Method (FGSM) adversarial attack on the trained Traffic Sign Classifier.


## Usage

### Training the Traffic Sign Classifier

To train the Traffic Sign Classifier, use the following command:

`python train.py --dataset /path/to/dataset --model output/trafficsignnet.model --plot plot.png`


This will train the TrafficSignNet model on the provided dataset and save the trained model to `output/trafficsignnet.model`. A plot of the training history will also be saved as `plot.png`.

### Performing the FGSM Attack

To perform the FGSM attack on the trained Traffic Sign Classifier, use the following command:

`python fgsm.py --image https://example.com/path/to/test/image.jpg`


This will generate adversarial examples for the Traffic Sign Classifier using different epsilon values and display the corresponding images.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Downloads
Download the dataset [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

Download signnames.csv [here](https://raw.githubusercontent.com/udacity/CarND-Traffic-Sign-Classifier-Project/master/signnames.csv)

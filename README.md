# Fast Gradient Sign Method (FGSM) Attack Showcase

This project demonstrates the Fast Gradient Sign Method (FGSM) adversarial attack on two different classifiers: one trained on the CIFAR-10 dataset and another trained to detect road signs. The goal of this project is to showcase the vulnerability of deep learning models to adversarial examples and provide insights into potential defense mechanisms.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [CIFAR-10 Classifier](https://github.com/stackviolator/adversarial-ml-demo/blob/master/cifar/README.md#usage)
  - [Road Sign Classifier](https://github.com/stackviolator/adversarial-ml-demo/blob/master/roadsigns/README.md#usage)
- [License](#license)

## Overview

The Fast Gradient Sign Method (FGSM) is a popular adversarial attack method that generates adversarial examples by adding small perturbations to the input image. These perturbations are designed to maximize the loss of the classifier, resulting in misclassification. This project implements the FGSM attack on two different models:

1. **CIFAR-10 Classifier**: A deep learning model trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
2. **Road Sign Classifier**: A deep learning model trained to detect various road signs.

## Installation

To set up this project, follow these steps:

1. Clone the repository:
`git clone https://github.com/stackviolator/adversarial-ml-demo.git`

2. Navigate to the project directory:

`cd adverasrial-demo`


3. Install the required dependencies:

`pip install -r requirements.txt`


## Usage

### [CIFAR-10 Classifier Usage](https://github.com/stackviolator/adversarial-ml-demo/blob/master/cifar/README.md#usage)
### [Road Sign Classifier Usage](https://github.com/stackviolator/adversarial-ml-demo/blob/master/cifar/README.md#usage)

## License

This project is licensed under the MIT License - see the LICENSE file for details.


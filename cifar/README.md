# Image Classifier and FGSM Attack Demo

This repository contains a script for an image classifier and Fast Gradient Sign Method (FGSM) attack demo. The script trains a neural network on the CIFAR-10 dataset, tests the trained model, and demonstrates the effect of adversarial perturbations on the model.

## Dependencies

- Python 3.6 or higher
- torch
- torchvision
- matplotlib
- numpy
- PIL (Pillow)

## Usage

To use the script, you can run the following command:
`python main.py [arguments]`


### Arguments

- `-t`, `--train`: Train the network.
- `-c`, `--cuda`: Enable CUDA if available.
- `-p`, `--perturb`: Test adversarial perturbation.
- `--test`: Test the network.
- `-e`, `--epochs`: Number of epochs to train (default: 2).
- `-ep`, `--epilson`: Input path to specify custom epsilon values (default: ../data/epsilons).
- `-b`, `--batch-size`: Input batch size for training (default: 4).
- `-i`, `--image`: Load an image.
- `-o`, `--outfile`: Output file for the trained network (default: ../nets/net.pth).
- `--infile`: Input file for the trained network (default: ../nets/net.pth).

## Example

To train the network with the default settings, run:
`python main.py --train`

To test the trained network, run:
`python main.py --test`

To test adversarial perturbation, run:
`To test adversarial perturbation, run:`

To test the network with a specific image, run:
`python main.py --image path/to/image.jpg`

## Output
The script displays the classification results and, if the `--perturb` option is used, shows a plot of the accuracy vs. epsilon values for adversarial perturbation. It also saves the trained network weights in the specified output file.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

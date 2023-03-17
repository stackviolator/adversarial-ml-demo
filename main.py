import argparse
import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import Net
import Unnormalize
from PIL import Image

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    epsilons = [0, .05, .1, .15]
    # epsilons = [0, .05, .1, .15, .2, .25, .3]
    # epsilons = [0, .005, .01, .015, .02, .025, .03]
    # epsilons = [0, .005, .01, .015]
    accuracies = []
    examples = []

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train the network')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='enable cuda')
    parser.add_argument('-p', '--perturb', action='store_true', default=False, help='test adversarial perturbation')
    parser.add_argument('--test', action='store_true', help='test the network')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='input batch size for training (default: 4)')
    parser.add_argument('-i', '--image', type=str, help='load an image')
    parser.add_argument('-o', '--outfile', default='nets/net.pth', help='output file for the trained network (default: nets/net.pth)')
    parser.add_argument('--infile', default='nets/net.pth', help='input file for the trained network (default: nets/net.pth)')

    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize the pics (bad for displaying w matplot)
         ]
    )

    batch_size = args.batch_size

    # Define datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    classes = {
        0 : 'plane',
        1 : 'car',
        2 :'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck'
    }

    # Enable CUDA if available
    if args.cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device != "cpu":
            print(f"Using CUDA device {device}")
    else:
        device = torch.device('cpu')

    # Instantiate the network
    net = Net.Net(args.cuda, device)
    net.to(device)

    if args.train:
        net.train(args.epochs, trainloader)
        net.save(args.outfile)
    else:
        net.load_state_dict(torch.load(args.infile, map_location=device))

    if args.test:
        net.test(testloader)

    if args.image:
        # Load image from path and resize to 32x32
        img = Image.open(args.image).convert('RGB')
        img = img.resize((32, 32))

        # Define transform to tensor and noramlize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Convert image to tensor
        img = transform(img)
        prediction = net.test_image(img)
        print( f"Prediction: {classes[prediction]}" )

    if args.perturb:
        for eps in epsilons:
            acc, ex = net.test_perturbed(testloader, eps)
            accuracies.append(acc)
            # Each index in examples is a tuple of (initial_prediction, final_prediction, image)
            examples.append(ex)

        '''
        # Show the epilson vs accuracy graph
        plt.figure(figsize=(5,5))
        plt.plot(epsilons, accuracies, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()
        '''

        # Need to unnormalize the images (normalized images = bad for displaying w matplot)
        un = Unnormalize.Unnormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Plot all images in examples
        x = 0
        plt.figure(figsize=(8,10))
        for i in range(len(examples)):
            for j in range(len(examples[i])):
                x += 1
                plt.subplot(len(epsilons),len(examples[0]), x)
                init_pred, final_pred, img_list = examples[i][j]
                for img in img_list:
                    img = un(torch.from_numpy(img))
                    img = img.T
                    plt.imshow(img)
                    plt.title(f"Image with epsilon {epsilons[i]}")
        plt.tight_layout()
        plt.show()

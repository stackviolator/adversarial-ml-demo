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
    accuracies = []
    examples = []
    epsilons = []

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train the network')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='enable cuda')
    parser.add_argument('-p', '--perturb', action='store_true', default=False, help='test adversarial perturbation')
    parser.add_argument('--test', action='store_true', help='test the network')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='input batch size for training (default: 4)')
    parser.add_argument('-i', '--image', type=str, help='load an image')
    parser.add_argument('-ep', '--epsilon', type=str, default="../data/epsilons", help='load custom epsilon values from a file')
    parser.add_argument('-o', '--outfile', default='../nets/net.pth', help='output file for the trained network (default: ../nets/net.pth)')
    parser.add_argument('--infile', default='../nets/net.pth', help='input file for the trained network (default: ../nets/net.pth)')

    args = parser.parse_args()

    # Add custom epsilon values
    with open(args.epsilon, 'r') as f:
        for line in f:
            # "#" is used as a comment
            if "#" not in line:
                epsilons.append(float(line))
    epsilons.append(float(1))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize the pics (bad for displaying w matplot)
         ]
    )

    batch_size = args.batch_size

    # Define training set and loader
    if args.train:
        trainset = torchvision.datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

    # Define test set and loader
    if args.test or args.perturb:
        testset = torchvision.datasets.CIFAR10(
            root='../data',
            train=False,
            download=True,
            transform=transform
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
            acc = net.test_perturbed(testloader, eps)
            accuracies.append(acc / 100)

        # Show the epilson vs accuracy graph
        plt.figure(figsize=(5,5))
        # Last epsilon is 1, we don't want to plot it
        plt.plot(epsilons[:-1], accuracies[:-1], "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        # Last epsilon is 1, we don't want to plot it, so we use epsilons[-2]
        # Step by the second epsilon value
        plt.xticks(np.arange(epsilons[0], epsilons[-2] + epsilons[1], step=float(epsilons[1])))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()


        # Plot first batch
        # Need to unnormalize the images (normalized images = bad for displaying w matplot)
        un = Unnormalize.Unnormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # Rows = epsilon level, column = image
        p_images, init_pred, final_pred, labels = net.get_perturbed_images(testloader, epsilons)

        x = 0
        plt.figure(figsize=(8,10))
        for row in range(len(p_images)):
            for col in range(p_images[row].shape[0]):
                img = p_images[row][col]
                x += 1
                plt.subplot(len(epsilons), 4, x)
                plt.xticks([], [])
                plt.yticks([], [])
                if (img == p_images[row][0]).all():
                    plt.ylabel("Eps: {}".format(epsilons[row]), fontsize=14)
                img = un(torch.from_numpy(img))
                img = img.permute(*torch.arange(img.dim() - 1, -1, -1))
                img = np.rot90(img)
                img = np.rot90(img)
                img = np.rot90(img)
                plt.title(f"Truth: {classes[int(labels[col].item())]}\nGuess: {classes[int(init_pred[0][col].item())]} -> {classes[int(final_pred[row][col].item())]}")
                plt.imshow(img)
        plt.tight_layout()
        plt.show()

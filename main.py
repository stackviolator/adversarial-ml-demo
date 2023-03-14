import argparse
import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import CNN

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train the network')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='enable cuda')
    parser.add_argument('--test', action='store_true', help='test the network')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='input batch size for training (default: 4)')
    parser.add_argument('-o', '--outfile', default='nets/net.pth', help='output file for the trained network (default: nets/net.pth)')
    parser.add_argument('-i', '--infile', default='nets/net.pth', help='input file for the trained network (default: nets/net.pth)')

    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Enable CUDA if available
    if args.cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Instantiate the network
    net = CNN.Net()
    net.to(device)

    if (args.train):
        net.train(args.epochs, trainloader)
        net.save(args.outfile)
    else:
        net.load_state_dict(torch.load(args.infile))

    if (args.test):
        net.test(testloader)

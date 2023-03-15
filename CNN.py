import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, cuda, device):
        super().__init__()
        self.cuda = cuda
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, epochs, trainloader):
        torch.backends.cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            print("Epoch: ", epoch + 1, "/", epochs)
            running_loss = 0.001
            for i, data in enumerate(trainloader, 0):
                if self.cuda:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    inputs, labels = data

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999 :
                    print(f"[{epoch + i}, {i + 1:5d} loss: {running_loss / 2000:.3f}]")
                    running_loss = 0.0

        print("Finished Training")

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def test(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                if self.cuda:
                    images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the 10k test images: {100 * correct // total}%")


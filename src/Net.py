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
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
            print("Epoch: ", epoch + 1, "/", epochs)
            running_loss = 0.001
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs.requires_grad = True

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

    def test_image(self, image):
        with torch.no_grad():
            image = image.to(self.device)
            '''
            ** BAD CODE **
            Conv2d expected a tensor of 4 bundled images, but we only have 1
            So we create a tensor of 4 images, all the same as the original
            '''
            img_list = image.tolist()
            image = torch.Tensor([img_list, img_list, img_list, img_list])
            output = self(image)
            _, predicted = torch.max(output.data, 1)
            # All the predictions are the same, so we only need the first one
            predicted = predicted.tolist()[0]
            return predicted

    def test(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                """
                Predicted is a tensor of guesses, labels is a tensor is ground truths
                Compare the tensors, result of (predicted == labels) is a tensor of bools,
                reflecting which of the predicted values were correct (represented as True)
                True = 1, False = -1, therefore a sum of the tensor is the amount of correct guesses
                """
                correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the 10k test images: {100 * correct // total}%")

    def fgsm_attack(self, images, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        # Here the attack
        p_images = images + (epsilon * sign_data_grad)
        # Clamp tensors to -1,1 range
        p_images = torch.clamp(p_images, -1, 1)
        return p_images

    def test_perturbed(self, testloader, epsilon):
        correct = 0
        total = 0
        adv_examples = []

        # Init the data
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs.requires_grad = True

            # Get the outputs and the prediction
            outputs = self(inputs)
            _, init_pred = torch.max(outputs.data, 1)

            total += labels.size(0)

            # Define the loss function and get the gradient for FGSM
            loss = F.nll_loss(outputs, labels)
            self.zero_grad()
            loss.backward()
            data_grad = inputs.grad

            # Define the perturbed data
            perturbed_inputs = self.fgsm_attack(inputs, epsilon, data_grad)

            # Get the new outputs and predictions
            outputs = self(perturbed_inputs)
            _, final_pred = torch.max(outputs.data, 1)

            # Keep stats on how many correct samples
            correct += (final_pred == labels).sum().item()

        final_acc = 100 * correct // total
        print(f"Accuracy of the network with Epsilon {epsilon} on the perturbed images: {100 * correct // total}%")

        return final_acc


    # TODO figure out if i want to skip the attack for images that are incorrectly labeled at e=0
    # The answer is yes but its just hard
    def get_perturbed_images(self, testloader, epsilons):
        # Rows = epsilon level, column = image
        p_images = []
        init_pred = []
        final_pred = []
        labels = []

        for data, target in testloader:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True

            output = self(data)
            _, pred = torch.max(output.data, 1)
            init_pred.append(pred)

            loss = F.nll_loss(output, target)
            self.zero_grad
            loss.backward()
            data_grad = data.grad

            for e in epsilons:
                perturbed_inputs = self.fgsm_attack(data, e, data_grad)
                p_images.append(perturbed_inputs.squeeze().detach().cpu().numpy())

                output = self(perturbed_inputs)
                _, pred = torch.max(output.data, 1)
                final_pred.append(pred)

            return p_images, init_pred, final_pred, target

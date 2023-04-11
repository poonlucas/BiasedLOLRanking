from torch import optim
from tqdm import tqdm
import numpy as np

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader

from dataloader import MatchDataset


# Building the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5 * 9, 20, bias=True)
        self.fc2 = nn.Linear(20, 10, bias=True)
        self.fc3 = nn.Linear(10, 5, bias=True)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten the feature matrix
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1)
        return x


# Train method
def train_model(model, train_loader, test_loader, optimizer, criterion, epochs):
    # training of the model
    print("Training the model ...\n")
    for epoch in range(epochs):

        # train the model
        model.train()
        train_loss = 0.0
        for input, target in tqdm(train_loader, total=len(train_loader)):
            # 1) zero the parameter gradients
            optimizer.zero_grad()
            # 2) forward + backward + optimize
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch + 1, train_loss))

        model.eval()
        correct = 0

        # Iterate through test data
        with torch.no_grad():
            for input, target in test_loader:
                # Make predictions
                output = model(input)
                predict_vals, predicted = torch.topk(output, k=5)

                # Track accuracy
                correct += (predicted == target).sum().item()

        # Calculate average loss and accuracy
        test_acc = correct / len(test_loader.dataset)

        # Print results
        print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(epoch + 1, 100. * test_acc))

    print("Finished Training")


if __name__ == '__main__':
    dataset = MatchDataset('data.csv')

    # Split the data into 80-20 train and test respectively, we want to make sure it is random, so we shuffle it
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # create dataloaders from train and test sets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, sampler=train_sampler)
    test_loader = DataLoader(dataset, sampler=test_sampler)

    # setup model, loss and optimizer
    model = Net()
    training_criterion = nn.MSELoss()  # use MSE here because we do regression
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.01)  # lr 0.001, weight_decay (L2 reg) 0.01

    train_model(model, train_loader, test_loader, optimizer, training_criterion, 10)


import torch
import torch.nn as nn # NN nodules, Linear, Conv2d, etc
import torch.optim as optim # All optimizers Adam, SDG
import torch.nn.functional as F
from torch.utils.data import DataLoader # Data management
import torchvision.datasets as datasets # Have standard datasets
import torchvision.transforms as transforms # Transformations on dataset .ToTensor()

# create fully connected Network
class NN(nn.Module):
    def __init__(self, input_size, n_class):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# model = NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x).shape) # gives 64x10

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784
n_class = 10
learning_rate = 0.001
batch_size = 64
epochs = 10

# data
train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# initialize
model = NN(input_size=input_size, n_class=n_class).to(device)

# loss and optim
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        # print(data) # MNIST = (64, 1, 28, 28) 64=examples, 1=balck&White, 28x28=height and width of img
        data = data.reshape(data.shape[0], -1) # make it (64, 784)

        # forward
        scores = model(data)
        loss = loss_function(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam
        optimizer.step()

# check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking on train data")
    else:
        print("Checking on test data")
    
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            data = data.reshape(data.shape[0], -1)

            scores = model(data) # dim 64x10
            _, prediction =  scores.max(1)
            num_correct += (prediction == targets).sum()
            num_samples += prediction.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    # model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
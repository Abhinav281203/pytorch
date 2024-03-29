import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class CNN(nn.Module):
    def __init__(self, in_channel = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x) 
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

# model = CNN()
# x = torch.randn(64, 1, 28, 28)
# print(model(x).shape)


#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
in_channel = 1
n_class = 10
learning_rate = 0.001
batch_size = 64
epochs = 1

# data
train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# initialize
model = CNN().to(device)

# loss and optim
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter('runs/MNIST/Tensorboard')

# train
step = 0
for epoch in range(epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = loss_function(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam
        optimizer.step()

        _, predictions = scores.max(1)
        num_correct = (predictions == targets).sum()
        running_training_accuracy = float(num_correct) / float(data.shape[0])

        writer.add_scalar("Training loss", loss, global_step=step)
        writer.add_scalar("Training Accurary", running_training_accuracy, global_step=step)
        step += 1
    
    mean_loss = sum(losses) / len(losses)
    print(f"loss at epoch {epoch} is {mean_loss:.5f}")

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

            scores = model(data) # dim 64x10
            _, prediction =  scores.max(1)
            num_correct += (prediction == targets).sum()
            num_samples += prediction.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    # model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
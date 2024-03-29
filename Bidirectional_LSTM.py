import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 28
sequence_length = 28
num_layer = 2
hidden_size = 256
n_class = 10
learning_rate = 0.001
batch_size = 64
epochs = 2

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, n_class):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, bidirectional=True)
        # N x time_sequence x features
        self.fc = nn.Linear(hidden_size * 2, n_class)

    def forward(self, x):
        h0 = torch.zeros(self.num_layer * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layer * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        # out = out.reshape(out.shape[0], -1)
        out = self.fc(out[:, -1, :])
        return out

# data
train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# initialize
model = BLSTM(input_size, hidden_size, num_layer, n_class).to(device)

# loss and optim
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device).squeeze(1) # to give nx28x28 instead
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

    mean_loss = sum(losses) / len(losses)
    print(f"loss at epoch {epoch} if {mean_loss:.5f}")

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
            data = data.to(device).squeeze(1)
            targets = targets.to(device)

            scores = model(data) # dim 64x10
            _, prediction =  scores.max(1)
            num_correct += (prediction == targets).sum()
            num_samples += prediction.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    # model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
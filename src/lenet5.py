"""LeNet-5 (tanh, average pooling) on MNIST from mnist_train.csv / mnist_test.csv (label, 784 pixels)."""
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class MNISTCSV(Dataset):
    def __init__(self, path):
        xs, ys = [], []
        with open(path, newline="") as f:
            for row in csv.reader(f):
                try:
                    label = int(float(row[0]))
                except ValueError:  # header row
                    continue
                ys.append(label)
                xs.append([float(v) for v in row[1:]])
        self.x = torch.tensor(xs, dtype=torch.float32) / 255.0
        self.y = torch.tensor(ys, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):  # pad 28x28 -> 32x32 as in the original LeNet-5
        return F.pad(self.x[i].view(1, 28, 28), (2, 2, 2, 2)), self.y[i]


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.pool1 = nn.Conv2d(1, 6, 5), nn.AvgPool2d(2, 2)
        self.conv2, self.pool2 = nn.Conv2d(6, 16, 5), nn.AvgPool2d(2, 2)
        self.fc1, self.fc2, self.fc3 = nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = x.flatten(1)
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


def accuracy(model, loader, device, optimizer=None, criterion=None):
    train = optimizer is not None
    model.train(train)
    correct = total = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if train:
                optimizer.zero_grad()
                criterion(logits, y).backward()
                optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(MNISTCSV("mnist_train.csv"), batch_size=128, shuffle=True)
    test_loader = DataLoader(MNISTCSV("mnist_test.csv"), batch_size=256)
    model = LeNet5().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1, 11):
        tr = accuracy(model, train_loader, device, opt, crit)
        te = accuracy(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Train Acc: {tr:.4f} | Test Acc: {te:.4f}")

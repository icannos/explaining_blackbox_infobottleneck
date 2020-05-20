import torch.nn as nn
import torch.nn.functional as F

class mnistConv(nn.Module):
    def __init__(self, device='cpu'):
        super(mnistConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5).to(device)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5).to(device)
        self.conv2_drop = nn.Dropout2d().to(device)
        self.fc1 = nn.Linear(320, 50).to(device)
        self.fc2 = nn.Linear(50, 10).to(device)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class cifar10(nn.Module):
    def __init__(self, device='cpu'):
        super(cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(6, 16, 5).to(device)  
        self.fc1 = nn.Linear(16 * 5 * 5, 120).to(device)
        self.fc2 = nn.Linear(120, 84).to(device)
        self.fc3 = nn.Linear(84, 10).to(device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
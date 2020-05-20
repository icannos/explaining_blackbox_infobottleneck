import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class convExplainer(nn.Module):
    def __init__(self, device='cpu'):
        super(convExplainer, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, padding=2).to(device)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=5, padding=2).to(device)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=5, padding=2).to(device)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=5, padding=2).to(device)
        self.conv5 = nn.Conv2d(1, 1, kernel_size=5, padding=2).to(device)


    def forward(self, x):
        return F.relu(self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))))

class denseExplainer(nn.Module):
    def __init__(self, device='cpu'):
        super(denseExplainer, self).__init__()

        self.f = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                               nn.Linear(64, 256), nn.Tanh(),
                                nn.Linear(256, 784), nn.Sigmoid()
                               ).to(device)


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        proba = self.f(x)
        return (x*proba).view(batch_size, 1, 28,28), proba

class convExplainerCifar(nn.Module):
    def __init__(self, device='cpu'):
        super(convExplainerCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, padding=2).to(device)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5, padding=2).to(device)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=5, padding=2).to(device)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=5, padding=2).to(device)
        self.conv5 = nn.Conv2d(3, 3, kernel_size=5, padding=2).to(device)

    def forward(self, x):
        score= F.relu(self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))))
        proba = nn.Sigmoid()(score)
        return x * proba, proba

class denseExplainerCifar(nn.Module):
    def __init__(self, device='cpu'):
        super(denseExplainerCifar, self).__init__()

        self.f = nn.Sequential(nn.Linear(3072, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                               nn.Linear(256, 256), nn.Tanh(),
                                nn.Linear(256, 3072), nn.ReLU()
                               ).to(device)


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        proba = self.f(x)
        return (x*proba).view(batch_size, 3, 32,32), proba


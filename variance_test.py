import torch
import torchvision
from models.cnn import mnistConv
from filter import denseExplainer, convExplainer
from models.explainer import ExplainerMnist
from matplotlib import pyplot as plt
from gaussian_reverse import ReverseDistributionMnist
from scipy.stats import multivariate_normal
import numpy as np

from torch.optim import Adam

distribution = ReverseDistributionMnist(device='cpu')
batch_size = 10

model = mnistConv('cpu')
model.load_state_dict(
    torch.load("logdir/mnist/checkpoints/best.pth", map_location=torch.device('cpu'))['model_state_dict'])

train_dataset = torchvision.datasets.MNIST('files/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST('files/', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ]))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#X, y = torch.randn((1,28,28)), None
X, y = train_dataset[8896]
#X, y = test_dataset[8896]

Xchap = X + 5*torch.randn_like(X)

proba = torch.exp(model(Xchap.unsqueeze(dim=0))).squeeze().detach().cpu().numpy()


plt.bar([i for i in range(10)], proba)
plt.show()



import torch
import torchvision
from models.cnn import mnistConv
from filter import denseExplainer, convExplainer
from models.explainer import ExplainerMnist
from matplotlib import pyplot as plt
from reverse_distribution import ReverseDistributionMnist

from torch.optim import Adam

batch_size = 256

distribution = ReverseDistributionMnist(device='cpu')

distribution.load_state_dict(torch.load('two_distrib.pth', map_location=torch.device('cpu')))

for i in range(10):
    y = torch.zeros((1, 10))
    y[0][i] = 1
    xchap, proba = distribution(y, 10)

    # print(torch.exp(model(xchap.view((1,1, 28,28)))))

    xchap = xchap.view(28, 28).detach().cpu().numpy()
    proba = proba.view(28, 28).detach().cpu().numpy()

    plt.imshow(xchap, cmap='gray')
    plt.title(f"Sample {i}")
    plt.show()

    plt.imshow(proba, cmap='gray')
    plt.title(f"proba {i}")
    plt.show()



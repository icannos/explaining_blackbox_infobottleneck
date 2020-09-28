import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from models.cnn import mnistConv
from filter import denseExplainer, convExplainer
from models.explainer import ExplainerMnist
from matplotlib import pyplot as plt
from gaussian_reverse import ReverseDistributionMnist
from scipy.stats import multivariate_normal
import numpy as np

T = np.array([-248948705.1920166, -1516184830.7631836, 1619340337.2020264, -423842794.04248047, 2366029285.7365723,
              165849372.12609863, -29383166039.57312, 1544927155.2685547, 2473356769.1170654, -754409545.8834229])

S = np.sum(np.abs(T))

print(T / S)

plt.bar([i for i in range(10)], T / S)
plt.show()

train_dataset = torchvision.datasets.MNIST('files/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))

X, y = train_dataset[852]

plt.imshow(X.reshape((28, 28)), cmap="gray")
plt.show()

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

model = mnistConv('cpu')
model.load_state_dict(
    torch.load("logdir/mnist/checkpoints/best.pth", map_location=torch.device('cpu'))['model_state_dict'])

distribution.load_state_dict(torch.load('gaussian_distrib.pth', map_location=torch.device('cpu')))

dist_params = []

batch_size = 16

for i in range(10):
    y = torch.zeros((1, 10))
    y[0][i] = 1
    xchap, mu, sigma = distribution(y)

    dist_params.append((mu, sigma))

    # print(torch.exp(model(xchap.view((1,1, 28,28)))))

    xchap = xchap.view(28, 28).detach().cpu().numpy()
    proba = mu.view(28, 28).detach().cpu().numpy()

    plt.imshow(xchap, cmap='gray')
    plt.title("Image")
    plt.show()

    # plt.imshow(proba, cmap='gray')
    # plt.title("Mu")
    # plt.show()

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

probas = []

for mu, sigma in dist_params:
    sigma = torch.matmul(sigma, torch.transpose(sigma, dim0=-2, dim1=-1))

    pseudo_logpdf = distribution.pseudo_log_pdf(X.view(1, 28*28), mu.squeeze(), sigma.squeeze())

    probas.append(pseudo_logpdf.detach().cpu().numpy())

print(probas)
print(y)

predicted = torch.exp(model(X.unsqueeze(dim=0))).squeeze().detach().cpu().numpy()

probas = np.array(probas)
S = np.sum(np.abs(probas))
probas = -probas / 10000

plt.bar([i for i in range(10)], predicted)
plt.show()

plt.bar([i for i in range(10)], probas)
plt.show()

X = X.detach().cpu().numpy()

plt.imshow(X.reshape((28, 28)), cmap="gray")
plt.show()


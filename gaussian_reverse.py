import torch
import torch.nn as nn
import torchvision
from models.cnn import mnistConv
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam


class ReverseDistributionMnist(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.device = device
        self.f = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU()).to(device)

        self.mu_f = nn.Sequential(nn.Linear(256, 784), nn.ReLU()).to(device)
        self.var_f = nn.Sequential(nn.Linear(256, 2048), nn.ReLU(), nn.Linear(2048, 76832), nn.ReLU()).to(device)

    def reparametrize(self, k, mu, sigma):
        x = torch.randn((k, 98, 1)).to(self.device)

        mul = torch.matmul(sigma, x)

        return mul + mu.unsqueeze(dim=2)

    def pseudo_log_pdf(self, x, mu, sigma):
        pinv = torch.pinverse(sigma)
        eig, _ = torch.eig(2 * np.pi * sigma.squeeze(), eigenvectors=False)
        eig = eig[:, 0]

        eig = eig[eig > 1E-8]

        pseudo_logdet = torch.sum(torch.log(eig), dim=-1)

        T = torch.matmul(pinv, (x - mu).unsqueeze(dim=2))

        T = torch.matmul(x - mu, T.squeeze())

        return (-(1 / 2) * pseudo_logdet - (1 / 2) * T).squeeze()

    def train_forward(self, x, batch_size):
        latent = self.f(x)

        mu = self.mu_f(latent)
        sigma = self.var_f(latent).view((-1, 784, 98))

        samples = []
        for i in range(10):
            samples.append(self.reparametrize(batch_size, mu[i].unsqueeze(dim=0), sigma[i].unsqueeze(dim=0)))

        samples = torch.cat(samples, dim=0).squeeze()

        return samples, mu, torch.matmul(sigma, torch.transpose(sigma, dim0=-2, dim1=-1))

    def forward(self, x):
        latent = self.f(x)

        mu = self.mu_f(latent)
        sigma = self.var_f(latent).view((-1, 784, 98))

        return self.reparametrize(mu.shape[0], mu, sigma), mu, torch.matmul(sigma, torch.transpose(sigma, dim0=-2, dim1=-1))


def xlogx(x):
    mask = (x > 1E-3).float()
    return (x * torch.log(x + 1E-7)) * mask


if __name__ == "__main__":
    batch_size = 1000

    model = mnistConv('cuda')
    distribution = ReverseDistributionMnist(device='cuda')

    model.load_state_dict(
        torch.load("logdir/mnist/checkpoints/best.pth", map_location=torch.device('cuda'))['model_state_dict'])

    optimizer = Adam(params=list(distribution.parameters()))

    inp = torch.Tensor([[i] for i in range(10)]).long()
    inp = inp.to('cuda')

    y = torch.Tensor([i for i in range(10) for _ in range(batch_size)]).long().to('cuda')

    for e in range(50):
        print(e)

        y_onehot = torch.zeros(10, 10).to('cuda')
        y_onehot.scatter_(1, inp, 1)

        xchap, mu, sigma = distribution.train_forward(y_onehot, batch_size)
        print(xchap.shape)
        xchap = xchap.view(-1, 1, 28, 28)
        output = model(xchap)

        entropies = []
        for i in range(10):
            eig, _ = torch.eig(2 * np.pi * sigma[i].squeeze(), eigenvectors=True)
            eig = eig[:, 0]

            eig = eig[eig > 1E-8]
            pseudo_logdet = torch.sum(torch.log(eig), dim=-1)

            entropies.append(pseudo_logdet)

        p = torch.exp(output)

        entropy = torch.stack(entropies).mean()

        loss = F.nll_loss(output, y.squeeze()) - entropy

        acc = (torch.argmax(p, dim=1) == y.squeeze()).float().mean()

        print(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss)

    torch.save(distribution.state_dict(), 'gaussian_distrib.pth')

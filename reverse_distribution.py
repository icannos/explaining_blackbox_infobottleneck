import torch
import torch.nn as nn
import torchvision
from models.cnn import mnistConv
from matplotlib import pyplot as plt
import torch.nn.functional as F

from torch.optim import Adam

class ReverseDistributionMnist(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.device = device
        self.f = nn.Sequential(nn.Linear(10, 128),  nn.ReLU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 784), nn.ReLU()).to(device)

    def reparametrize(self, proba, k, temperature=5):
        # - log( - log(U) ) ~ Gumbel

        #img = []
        img = 0
        for _ in range(k):
            gumbel_noise = - torch.log(-torch.log(torch.rand_like(proba))).to(self.device)

            x = (torch.log(proba) + gumbel_noise) / temperature

            sample = nn.functional.softmax(x, dim=-1)
            #img.append(sample)
            img += sample

        #img = torch.stack(img, dim=-1)
        #img, indices = torch.max(img, dim=-1)

        return img

    def forward(self, yonehot, k):
        proba = self.f(yonehot)
        return self.reparametrize(proba, k=k), proba


def xlogx(x):
    mask = (x > 1E-3).float()
    return (x * torch.log(x + 1E-7))*mask

if __name__ == "__main__":
    batch_size = 512

    model = mnistConv('cuda')
    distribution = ReverseDistributionMnist(device='cuda')

    model.load_state_dict(torch.load("logdir/mnist/checkpoints/best.pth", map_location=torch.device('cuda'))['model_state_dict'])

    optimizer = Adam(params=list(distribution.parameters()))

    for e in range(100):

        y = torch.randint(0, 10, (batch_size,1))
        y = y.to('cuda')

        y_onehot = torch.zeros(batch_size, 10).to('cuda')
        y_onehot.scatter_(1, y, 1)

        xchap, proba = distribution(y_onehot, 50)
        xchap = xchap.view(batch_size, 1, 28,28)
        output = model(xchap)

        p = torch.exp(output)

        img_entropy = - torch.sum(xlogx(proba).view(batch_size, -1), dim=1).mean()

        loss = F.nll_loss(output, y.squeeze(), reduction="mean") - 0.001 * img_entropy
        #loss = F.l1_loss(output, y_onehot, reduction="mean")

        acc = (torch.argmax(p, dim=1) == y.squeeze().to('cuda')).float().mean()

        print(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss)

    torch.save(distribution.state_dict(), 'two_distrib.pth')











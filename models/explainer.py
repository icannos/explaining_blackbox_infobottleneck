import torch
import torch.nn as nn

class ExplainerMnist(nn.Module):

    def __init__(self, device='cpu'):
        super(ExplainerMnist, self).__init__()
        self.device = device

        self.f = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                               nn.Linear(64, 256), nn.Tanh(),
                                nn.Linear(256, 784), nn.Sigmoid()
                               ).to(device)

    def reparametrize(self, proba, k, temperature=0.1):
        # - log( - log(U) ) ~ Gumbel

        mask = []
        for _ in range(k):
            gumbel_noise = - torch.log(-torch.log(torch.rand_like(proba))).to(self.device)

            x = (torch.log(proba) + gumbel_noise) / temperature

            sample = nn.functional.softmax(x, dim=-1)
            mask.append(sample)

        mask = torch.stack(mask, dim=-1)
        mask, indices = torch.max(mask, dim=-1)

        return mask

    def make_mask(self, x, k):
        batch_size = x.shape[0]

        proba = self.f(x.view(batch_size, -1))
        mask = self.reparametrize(proba, k)

        return mask, proba

    def forward(self, x, k):
        mask, proba = self.make_mask(x, k)
        return x * mask.view_as(x), proba


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import mnistConv
from utils.projected_gradient_descent import projected_gradient_descent

from torch.optim import Adam
from advertorch.context import ctx_noparamgrad_and_eval

from advertorch.attacks import L2PGDAttack, LinfPGDAttack, L1PGDAttack

import argparse


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddUniformNoise(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size())

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ReverseDistributionMnist(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.device = device

        # Apprend une distribution de proba sur les pixels à allumer
        self.f = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 784),
                               nn.ReLU()).to(device)

    def reparametrize(self, proba, k, temperature=0.1):
        # - log( - log(U) ) ~ Gumbel

        # img = []
        img = 0
        for _ in range(k):
            gumbel_noise = - torch.log(-torch.log(torch.rand_like(proba))).to(self.device)

            x = (torch.log(proba) + gumbel_noise) / temperature

            sample = nn.functional.softmax(x, dim=-1)
            # img.append(sample)
            img += sample

        # img = torch.stack(img, dim=-1)
        # img, indices = torch.max(img, dim=-1)

        return img

    def forward(self, yonehot, k):
        proba = self.f(yonehot)
        return self.reparametrize(proba, k=k), proba


# Fonction pour garantir la stabilité numérique (0 * log(0) = 0)
def xlogx(x):
    mask = (x > 1E-3).float()
    return (x * torch.log(x + 1E-7)) * mask


device = 'cuda'
batch_size = 256

num_epoch = 1

# adv_trains = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.2, 1.4, 1.6]
adv_trains = [0.1, 1., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 70., 80., 100.]

stats_iter = 100

stats = {}

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)),
                                   AddUniformNoise()
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

# for attack_type in [L2PGDAttack, L1PGDAttack, LinfPGDAttack]:
for attack_type in [L1PGDAttack, L2PGDAttack]:
    print(attack_type.__name__)
    for adv_train in adv_trains:
        # logdir = f"trained_models/adv_train-l2{adv_train}"
        stats[attack_type.__name__ + "_" + str(adv_train)] = {}
        stats[attack_type.__name__ + "_" + str(adv_train)]['accs'] = []
        stats[attack_type.__name__ + "_" + str(adv_train)]['acc_generator'] = []
        stats[attack_type.__name__ + "_" + str(adv_train)]['proba_maps'] = []
        stats[attack_type.__name__ + "_" + str(adv_train)]['models'] = []

        for iter in range(stats_iter):

            model = mnistConv(device=device).to(device)
            # We use categorical cross entropy but on log softmax
            criterion = nn.NLLLoss()
            # Adam as optimizer since it is the most used optimizer. We'll see later on if other method works better since we know
            # that adam do not generalize well on image processing problem
            optimizer = Adam(model.parameters())

            adversary = attack_type(
                model, loss_fn=nn.NLLLoss(reduction="mean"), eps=adv_train,
                nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=False)

            for e in range(num_epoch):
                print(e)
                for x, y in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    bsize = y.shape[0]

                    with ctx_noparamgrad_and_eval(model):
                        adv_targeted = adversary.perturb(x, y)

                    optimizer.zero_grad()

                    loss = criterion(model(x), y) + criterion(model(adv_targeted), y)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                accs = []
                for x, y in test_loader:
                    preds = torch.argmax(model(x.to(device)), dim=1)
                    acc = (preds == y.to(device)).float().mean()

                    accs.append(acc)

                accs = torch.stack(accs)
                accs = accs.mean()

            stats[attack_type.__name__ + "_" + str(adv_train)]['accs'].append(accs.cpu().detach().numpy())
            stats[attack_type.__name__ + "_" + str(adv_train)]['models'].append(model.state_dict())

            # Modèle pour apprendre la distribution
            distribution = ReverseDistributionMnist(device='cuda')

            # optimizer
            optimizer = Adam(params=list(distribution.parameters()))

            # Entraînement du modèle génératif

            for e in range(100):
                print(e)
                y = torch.randint(0, 10, (batch_size, 1))
                y = y.to('cuda')

                y_onehot = torch.zeros(batch_size, 10).to('cuda')
                y_onehot.scatter_(1, y, 1)

                xchap, proba = distribution(y_onehot, 500)
                xchap = xchap.view(batch_size, 1, 28, 28)
                output = model(xchap)

                p = torch.exp(output)

                img_entropy = - torch.sum(xlogx(proba).view(batch_size, -1), dim=1).mean()

                loss = ((1 - (torch.argmax(output, dim=1) == y.squeeze()).float()) * F.nll_loss(output, y.squeeze(),
                                                                                                reduction='none')).mean() - 0.01 * img_entropy
                # loss = F.l1_loss(output, y_onehot, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f'{loss}')

            acc = (torch.argmax(p, dim=1) == y.squeeze().to('cuda')).float().mean()

            y = torch.zeros((10, 10))

            for i in range(10):
                y[i][i] = 1

            xchap, proba = distribution(y.to('cuda'), 10)

            stats[attack_type.__name__ + "_" + str(adv_train)]['acc_generator'].append(acc.detach().cpu().numpy())
            stats[attack_type.__name__ + "_" + str(adv_train)]['proba_maps'].append(proba.detach().cpu().numpy())

import pickle as pk

pk.dump(stats, open('tmp/stats_l1l2_fine_wholegenerator.dat', 'wb'), protocol=pk.HIGHEST_PROTOCOL)

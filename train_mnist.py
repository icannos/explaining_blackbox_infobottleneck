import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import mnistConv
from utils.projected_gradient_descent import projected_gradient_descent

from torch.optim import Adam
from advertorch.context import ctx_noparamgrad_and_eval

from catalyst.dl import SupervisedRunner, AccuracyCallback, AUCCallback, ConfusionMatrixCallback, \
    PrecisionRecallF1ScoreCallback

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




device = 'cuda'
batch_size = 256

num_epoch = 10

#stds = [0.1, 0.5, 1, 1.5, 2, 2.5,3]
#adv_trains = [0.15, 0.5, 1., 5.,10., 20.]
adv_trains = [0]

# for std in stds:
for adv_train in adv_trains:
    logdir = f"trained_models/base.dat"

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


    model = mnistConv(device=device).to(device)
    # We use categorical cross entropy but on log softmax
    criterion = nn.NLLLoss()
    # Adam as optimizer since it is the most used optimizer. We'll see later on if other method works better since we know
    # that adam do not generalize well on image processing problem
    optimizer = Adam(model.parameters())

    from advertorch.attacks import L2PGDAttack

    # adversary = L2PGDAttack(
    #     model, loss_fn=nn.NLLLoss(reduction="mean"), eps=adv_train,
    #     nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    #     targeted=False)

    for e in range(num_epoch):
        print(e)
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            bsize = y.shape[0]

            #with ctx_noparamgrad_and_eval(model):
                #adv_targeted = adversary.perturb(x, y)

            optimizer.zero_grad()

            loss = criterion(model(x), y)# + criterion(model(adv_targeted), y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(loss)

    torch.save(model.state_dict(), logdir)






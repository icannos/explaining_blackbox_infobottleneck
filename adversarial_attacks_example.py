import torch
import torchvision
import torch.nn as nn
from models.cnn import mnistConv
from filter import denseExplainer, convExplainer
from models.explainer import ExplainerMnist
from matplotlib import pyplot as plt
import torch.nn.functional as F

from torch.optim import Adam
import numpy as np

from advertorch.attacks import L2PGDAttack, LinfPGDAttack, L1PGDAttack, MomentumIterativeAttack, LinfMomentumIterativeAttack, L2MomentumIterativeAttack

def xlogx(x):
    mask = (x > 1E-3).float()
    return (x * torch.log(x + 1E-7))*mask



batch_size = 4

def plot_attacks(data, batch_size):

    for attack_name,v in data.items():
        fig = plt.figure()

        num_params = len(v)

        fig, axs = plt.subplots(2*batch_size, num_params+1, figsize=(num_params+1, 2*batch_size))

        for k, (p, (x, x_adv, preds)) in enumerate(v.items()):
            for i in range(batch_size):
                axs[2*i, 0].imshow(x[i].reshape((28, 28)), cmap="gray")
                axs[2*i, 0].axis("off")
                if i == 0:
                    axs[0, 0].set_title("Original")

                axs[2 * i + 1, 0].bar([i for i in range(10)], preds[i])

                axs[2 * i + 1, 0].get_yaxis().set_visible(False)
                x_loc = [np.argmax(preds[i])]

                axs[2 * i + 1, 0].set_xticks(x_loc)
            break

        for k, (p, (x, x_adv, preds)) in enumerate(v.items()):
            for i in range(batch_size):
                axs[2*i, k+1].imshow(x_adv[i].reshape((28,28)), cmap="gray")
                axs[2*i, k+1].axis("off")
                if i == 0:
                    axs[2*i, k+1].set_title(p)

                axs[2 * i+1, k + 1].bar([i for i in range(10)], preds[i])

                axs[2 * i + 1, k+1].get_yaxis().set_visible(False)
                x_loc = [np.argmax(preds[i])]

                axs[2 * i + 1, k+1].set_xticks(x_loc)

        #fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)
        fig.subplots_adjust(hspace=0.005)
        fig.tight_layout()


        fig.savefig("tmp/adversarial_example_" + attack_name+".png")

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

mnist_data = torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

test_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

model = mnistConv('cuda')

model.load_state_dict(torch.load("trained_models/base.dat", map_location=torch.device('cuda')))


params_sets = [[0.1, 1., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55.],
                [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.2, 1.4],
               [0.1, 1., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55.]]


attacks = [L2PGDAttack, LinfPGDAttack, L1PGDAttack]

data = {attack.__name__: {p:None for p in params_sets[k]} for k, attack in enumerate(attacks)}

for x,y in test_loader:
    x = x.to('cuda')
    y = y.to("cuda")
    for k,attack_type in enumerate([L2PGDAttack, LinfPGDAttack, L1PGDAttack]):

        for adv_train in params_sets[k]:
            adversary = attack_type(
                model, loss_fn=nn.NLLLoss(reduction="sum"), eps=adv_train,
                nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=False)

            adv_targeted = adversary.perturb(x, y)

            preds = torch.exp(model(adv_targeted))

            predsp = preds.detach().cpu().numpy()
            adv_targetedp = adv_targeted.detach().cpu().numpy()
            xp = x.detach().cpu().numpy()

            data[attack_type.__name__][adv_train] = (xp, adv_targetedp, predsp)

    break

plot_attacks(data, batch_size)

















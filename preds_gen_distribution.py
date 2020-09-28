

import torch
import torch.nn as nn
import torchvision
from models.cnn import mnistConv
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax

from models.explainer import ExplainerMnist
from torch.optim import Adam

import pickle as pk
from scipy.stats import wasserstein_distance

stats = pk.load(open('stats/stats_linf_fine.dat', 'rb'))

batch_size = 256

def log_gen_probability(log_proba_map, X):
    neg_proba = torch.log(1-torch.exp(log_proba_map))

    return (log_proba_map *X).sum(dim=1) #+ ((1 - X) * neg_proba).sum(dim=1)

test_dataset =     torchvision.datasets.MNIST('files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

N = len(stats)
# adv_trains = [0.1, 0.5, 1., 1.5, 2., 5., 10., 20., 30., 50.]
adv_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.2, 1.4, 1.6]
#adv_trains = [0.1, 1., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 70., 80., 100.]

#attack_names = ["L2PGDAttack", "L1PGDAttack"]
attack_names = ["LinfPGDAttack"]

for attack_name in attack_names:

    params_data = np.zeros((len(adv_trains), 100))

    accs_for_each_param = []
    for k, adv in enumerate(adv_trains):
        print(k)
        v = stats[f"{attack_name}_{str(adv)}"]

        data = np.asarray(v['proba_maps']) # t, digit

        accs = []

        for t in range(100):
            proba_maps = torch.Tensor(data[t]).to('cuda')
            targets = []
            preds = []

            for x,y in test_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                targets.append(y)
                probas_i = []
                for i in range(10):
                    proba_i = log_gen_probability(proba_maps[i], x.view(-1, 28 * 28))

                    probas_i.append(proba_i)
                probas_i = torch.stack(probas_i).t()
                preds.append(torch.argmax(probas_i, dim=1))

            preds = torch.cat(preds)
            targets = torch.cat(targets)
            accs.append((preds == targets).float().mean().cpu().detach().numpy())

        accs_for_each_param.append(accs)

    accs_for_each_param = np.asarray(accs_for_each_param)

    stds = np.std(accs_for_each_param, axis=1)
    means = np.mean(accs_for_each_param, axis=1)

    print(means)

    mean_up = means + stds
    mean_down = means - stds

    plt.plot(adv_trains, means)
    plt.fill_between(adv_trains, mean_up, mean_down, alpha=0.5)
    plt.xlabel('Ball sizes')
    plt.ylabel('Accuracy')

    plt.savefig(f"tmp/{attack_name}-gendistrib_pred_accuracy.png")
    plt.clf()


# for t in range(1):
#
#     for attack_name in attack_names:
#         fig, axs = plt.subplots(10, len(adv_trains), figsize=(10, 10))
#
#         for k, key in enumerate(adv_trains):
#             v = stats[f"{attack_name}_{str(key)}"]
#
#             data = np.asarray(v['proba_maps'])  # t, digit
#
#             maps = torch.Tensor(v['proba_maps'][t])
#
#             probas_maps = torch.Tensor(data[t])
#
#             for p in range(5):
#                 x = test_dataset[np.random.randint(0, 10000)][0]
#
#                 probas = []
#                 for i in range(10):
#                     proba_i = log_gen_probability(probas_maps[i], x.view(-1, 28 * 28))
#
#                     probas.append(proba_i)
#
#                 probas = torch.softmax(torch.stack(probas).t(), dim=1).cpu().detach().numpy()
#
#                 axs[2*p, k].imshow(x.detach().cpu().numpy().reshape((28,28)), cmap="gray")
#                 axs[2*p, k].axis('off')
#
#                 axs[2 * p+1, k].bar([i for i in range(10)], probas[0])
#                 axs[2 * p+1, k].get_yaxis().set_visible(False)
#                 x_loc = [np.argmax(probas[0])]
#
#                 axs[2 * p + 1, k].set_xticks(x_loc)
#
#
#         fig.subplots_adjust(hspace=0.01)
#         fig.tight_layout()
#         fig.savefig(f"tmp/{attack_name}_preds_gen_distribution_samples_{t}.png")







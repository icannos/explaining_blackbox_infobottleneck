# Quelques imports

import torch
import torch.nn as nn
import torchvision
from models.cnn import mnistConv
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax

from torch.optim import Adam

import pickle as pk
from scipy.stats import wasserstein_distance


filename = "tmp/stats_linf_fine_dense_sharp.dat"
stats = pk.load(open(filename, 'rb'))
for k in stats.keys():
    stats[k]['models'] = None

pk.dump(stats, open(filename + "small", 'wb'), protocol=pk.HIGHEST_PROTOCOL)
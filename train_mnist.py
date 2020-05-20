import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import mnistConv

from torch.optim import Adam

from catalyst.dl import SupervisedRunner, AccuracyCallback, AUCCallback, ConfusionMatrixCallback, \
    PrecisionRecallF1ScoreCallback


batch_size = 256
logdir = "logdir/mnist/"
num_epoch = 5

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
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

loaders = {"train": train_loader, "valid": test_loader}

model = mnistConv(device='cuda').to('cuda')
# We use categorical cross entropy but on log softmax
criterion = nn.NLLLoss()
# Adam as optimizer since it is the most used optimizer. We'll see later on if other method works better since we know
# that adam do not generalize well on image processing problem
optimizer = Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# We use Catalyst to ease the training process and have well saved record in tensorboard
runner = SupervisedRunner(device='cuda')

# We run the training loop with Accuracy metric and confusion matrix to get recall data.
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epoch,
    verbose=True,
    callbacks=[AccuracyCallback(),
               ConfusionMatrixCallback(num_classes=10), ]
)


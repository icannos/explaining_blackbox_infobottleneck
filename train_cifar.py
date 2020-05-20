import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import cifar10
import torchvision.transforms as transforms

from torch.optim import Adam

from catalyst.dl import SupervisedRunner, AccuracyCallback, AUCCallback, ConfusionMatrixCallback, \
    PrecisionRecallF1ScoreCallback


batch_size = 16
logdir = "logdir/cifar/"
num_epoch = 20

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./files', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./files', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

loaders = {"train": train_loader, "valid": test_loader}

model = cifar10(device='cuda').to('cuda')
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


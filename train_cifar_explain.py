import torch
import torchvision
from models.cnn import cifar10
from filter import denseExplainerCifar, convExplainerCifar
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.optim import Adam

def xlogx(x):
    mask = (x > 1E-3).float()
    return (x * torch.log(x + 1E-7))*mask

batch_size = 256
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

model = cifar10('cuda')
explainer = denseExplainerCifar('cuda')

model.load_state_dict(torch.load("logdir/cifar/checkpoints/best.pth", map_location=torch.device('cuda'))['model_state_dict'])

optimizer = Adam(params=list(explainer.parameters()))

for x, y in train_loader:
    x = x.to('cuda')
    xchap, proba = explainer(x)

    output = model(xchap)
    p = torch.exp(output)
    model_output = model(x)

    #loss = F.mse_loss(output, model_output) + 0.01*torch.sum(xchap, dim=1).mean()
    img_entropy = - torch.sum(xlogx(proba.view(batch_size, -1)), dim=1).mean()

    loss = 100*F.nll_loss(output, torch.argmax(model_output, dim=1), reduction="mean") + 5*img_entropy + 0.5*torch.sum(proba.view(batch_size, -1), dim=1).mean()

    acc = (torch.argmax(p, dim=1) == y.to('cuda')).float().mean()

    print(acc)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(loss)

torch.save(explainer.state_dict(), 'explainer_cifar.pth')










import torch
import torchvision
from models.cnn import mnistConv, cifar10
from filter import denseExplainer, convExplainer, convExplainerCifar, denseExplainerCifar
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

from torch.optim import Adam

batch_size = 256


transform = transforms.Compose(
    [transforms.ToTensor(),])

trainset = torchvision.datasets.CIFAR10(root='./files', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./files', train=False,
                                       download=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

model = cifar10('cpu')
explainer = denseExplainerCifar('cpu')

model.load_state_dict(torch.load("logdir/cifar/checkpoints/best.pth", map_location=torch.device('cpu'))['model_state_dict'])

explainer.load_state_dict(torch.load('explainer_cifar.pth', map_location=torch.device('cpu')))


x = torch.unsqueeze(trainset[896][0],0).to('cpu')
xchapp, proba = explainer(x)
xchap = torch.squeeze(xchapp).transpose(0,2).cpu().detach().numpy()

print(xchapp.shape)

plt.imshow(xchap)
plt.show()

proba_numpy = proba.view((32,32, 3)).cpu().detach().numpy()
plt.imshow(proba_numpy)
plt.show()

print(x.shape)
plt.imshow(torch.squeeze(x).transpose(0,2).cpu().detach().numpy())
plt.show()

print(torch.exp(model(xchapp)))
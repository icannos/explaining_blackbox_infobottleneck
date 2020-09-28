import torch
import torchvision
from models.cnn import mnistConv
from filter import denseExplainer, convExplainer
from models.explainer import ExplainerMnistNegative
from matplotlib import pyplot as plt
import torch.nn.functional as F

from torch.optim import Adam

def xlogx(x):
    mask = (x > 1E-3).float()
    return (x * torch.log(x + 1E-7))*mask

batch_size = 256

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
explainer = ExplainerMnistNegative('cuda')

model.load_state_dict(torch.load("logdir/mnist/checkpoints/best.pth", map_location=torch.device('cuda'))['model_state_dict'])

optimizer = Adam(params=list(explainer.parameters()))

for e in range(5):
    for x, y in train_loader:
        x = x.to('cuda')
        xchap, proba_pos, proba_neg = explainer(x, k=20, n=500)


        output = model(xchap)
        p = torch.exp(output)
        model_output = model(x)


        #loss = F.mse_loss(output, model_output) + 0.01*torch.sum(xchap, dim=1).mean()

        loss = F.nll_loss(output, torch.argmax(model_output, dim=1), reduction="mean")
        #loss = F.mse_loss(output, model_output, reduction="mean")

        acc = (torch.argmax(p, dim=1) == y.to('cuda')).float().mean()

        print(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss)

torch.save(explainer.state_dict(), 'explainer.pth')










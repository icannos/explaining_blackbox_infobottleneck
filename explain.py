import torch
import torchvision
from models.cnn import mnistConv
from filter import denseExplainer, convExplainer
from models.explainer import ExplainerMnist
from matplotlib import pyplot as plt

from torch.optim import Adam

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

model = mnistConv('cpu')
explainer = ExplainerMnist('cpu')

model.load_state_dict(torch.load("logdir/mnist/checkpoints/best.pth", map_location=torch.device('cpu'))['model_state_dict'])

explainer.load_state_dict(torch.load('explainer.pth', map_location=torch.device('cpu')))


x = torch.unsqueeze(mnist_data[1258][0],0).to('cpu')
xchapp, proba = explainer(x, k=20)
mask, proba = explainer.make_mask(x, k=20)
mask = mask.view((28,28))

maskx = torch.stack([mask, x.squeeze(), torch.zeros_like(mask)], dim=-1)
maskx = maskx.cpu().detach().numpy()

xchap = torch.squeeze(xchapp).cpu().detach().numpy()
mask = torch.squeeze(mask).cpu().detach().numpy()

plt.imshow(xchap, cmap='gray')
plt.title("Masked input")
plt.show()

plt.imshow(maskx)
plt.title("Mask in red and original input in green")
plt.show()

proba_numpy = proba.view((28,28)).cpu().detach().numpy()
plt.imshow(proba_numpy, cmap='gray')
plt.title("Distribution of probability to select pixel in the mask")
plt.show()

plt.imshow(torch.squeeze(x).cpu().detach().numpy(), cmap='gray')
plt.show()

print(torch.exp(model(xchapp)))
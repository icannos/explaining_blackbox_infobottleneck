import torch
import torch.nn as nn


def determinant_sdp_matrix(X):
    U = torch.cholesky(X).diagonal(dim1=1, dim2=2)
    det = torch.prod(U, dim=1)

    return det



def kl_div_multivariate_gaussian(mu1, sigma1, mu2, sigma2):
    inv_sigma2 = torch.cholesky_inverse(sigma2)
    mu1mu2 = mu1 - mu2
    k = mu1.shape[1]

    t1 = (inv_sigma2 @ sigma2).diagonal(dim1=-2, dim2=-1).sum()
    tmp = (inv_sigma2 @ mu1mu2.unsqueeze(dim=2)).squeeze()

    t2 = (mu1mu2 * tmp).sum(dim=1)

    t3 = torch.log(torch.det(sigma2) / torch.det(sigma1))

    return ((1/2) * (t1 + t2 +t3 - k)).squeeze()


def square_root_matrix_sdp(X):
    torch.svd()

def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    d = torch.norm(mu1 - mu2, dim=1)

    sigma1 + sigma2 - 2 * ()



sigma1 =  10* torch.rand((2, 5,5)) + 10
sigma1 = sigma1 @ sigma1.transpose(1,2)

sigma2 =  10* torch.rand((2, 5,5)) + 10
sigma2 = sigma2 @ sigma2.transpose(1,2)

mu1 = torch.zeros(2,5)
mu2 = torch.zeros(2,5)

sigma1.requires_grad_(True)
sigma2.requires_grad_(True)
mu1.requires_grad_(True)
mu2.requires_grad_(True)


print(torch.cholesky(sigma1)[0])







import torch
import torch.nn as nn
from torch.distributions.transforms import Transform

class Sequential(torch.nn.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for computing the output of
    the function alongside with the log-det-Jacobian of such transformation.
    """

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """

        log_det_jacobian = 0.
        for i, module in enumerate(self._modules.values()):
            inputs, log_det_jacobian_ = module(inputs)
            log_det_jacobian = log_det_jacobian + log_det_jacobian_
        return inputs, log_det_jacobian

class TrainableBijector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass

    def inverse_forward(self, *args, **kwargs):
        pass

    def log_det_jacobian(self, *args, **kwargs):
        pass

class StrictlyDominantLinearTransform(nn.Module):
    def __init__(self, in_features: int, out_features: int, dim: int, bias: bool = True):
        super().__init__()
        




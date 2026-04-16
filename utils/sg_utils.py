import torch
import torch.nn.functional as F


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


def normalize_sg_axis(axis: torch.Tensor) -> torch.Tensor:
    return F.normalize(axis, dim=-1)


def eval_sg(viewdirs: torch.Tensor, axis: torch.Tensor, sharpness: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    viewdirs = F.normalize(viewdirs, dim=-1)
    axis = normalize_sg_axis(axis)
    dot = torch.sum(viewdirs[:, None, :] * axis, dim=-1, keepdim=True)
    return torch.sum(amplitude * torch.exp(sharpness * (dot - 1.0)), dim=1)

import torch
import torch.nn as nn
from torchvision import models


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


_ARCH_WEIGHTS = {
    "resnet50": models.ResNet50_Weights,
    "resnet18": models.ResNet18_Weights,
    "resnet34": models.ResNet34_Weights,
    "resnet101": models.ResNet101_Weights,
}


def get_model(cfg, device):
    """Build a ResNet with ImageNet-pretrained weights, wrapped in a Normalize layer.

    The Normalize layer lets attack code treat inputs as raw [0, 1] tensors.
    """
    arch = cfg["model"]["arch"]
    weights = getattr(_ARCH_WEIGHTS[arch], cfg["model"]["weights"])
    base = getattr(models, arch)(weights=weights)

    mean = cfg["normalization"]["mean"]
    std = cfg["normalization"]["std"]
    model = nn.Sequential(Normalize(mean, std), base).to(device)
    model.eval()
    return model

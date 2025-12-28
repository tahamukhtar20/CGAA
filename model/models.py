import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks import PGD

def get_model(device, mu, std):
    m = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    m = m.to(device).eval()
    return NormWrapper(m, mu, std)

class NormWrapper(nn.Module):
    def __init__(self, model, mu, std):
        super().__init__()
        self.model = model
        self.register_buffer('mu', mu)
        self.register_buffer('std', std)
    def forward(self, x): 
        return self.model((x - self.mu) / self.std)

class CGAA(PGD):
    def __init__(self, model, v_C, lam, device, target_layer="model.layer3", **kwargs):
        super().__init__(model, **kwargs)
        self.v_C = v_C.to(device)
        self.lam = lam
        self.device = device
        self.target_layer = target_layer
        self.phi_x = None

    def forward(self, x, y):
        x, y = x.clone().detach().to(self.device), y.to(self.device)
        x_adv = x.clone().detach().requires_grad_(True)
        
        layer = dict(self.model.named_modules())[self.target_layer]
        def hook(m, i, o): self.phi_x = o.flatten(start_dim=1)
        h = layer.register_forward_hook(hook)

        for _ in range(self.steps):
            out = self.model(x_adv)
            S_C = torch.matmul(self.phi_x, self.v_C).mean()
            loss = F.cross_entropy(out, y) + (self.lam * S_C)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
        h.remove()
        return x_adv

import os
import random

import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_phi(model, loader, device, layer_name='model.layer3', flatten=True):
    Phi = []
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer '{layer_name}' not found.")
    
    target_layer = modules[layer_name]
    def hook(m, i, o):
        feat = o.detach()
        if flatten: feat = feat.flatten(start_dim=1)
        Phi.append(feat.cpu())

    h = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        for x, _ in loader: model(x.to(device))
    h.remove()
    return torch.cat(Phi)

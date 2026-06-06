"""Concept Activation Vector (CAV) training.

A CAV is the normal of a linear classifier trained to separate "concept-positive"
activations from "concept-negative" activations at a given layer. We pool spatial
dims and train an SGD classifier on penultimate-style features.
"""
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def _pooled_activations(model, loader, layer_name, device):
    acts = []
    layer_mod = dict(model[1].named_modules())[layer_name]

    def hook(_m, _i, o):
        acts.append(o.mean(dim=[2, 3]).detach().cpu().numpy())

    handle = layer_mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for x, _ in loader:
                model(x.to(device))
    finally:
        handle.remove()
    return np.concatenate(acts)


def train_cav(model, pos_loader, neg_loader, layer_name, device, cfg):
    """Train a linear concept classifier; return (cav_tensor, val_accuracy)."""
    act_p = _pooled_activations(model, pos_loader, layer_name, device)
    act_n = _pooled_activations(model, neg_loader, layer_name, device)

    X = np.concatenate([act_p, act_n])
    y = np.concatenate([np.ones(len(act_p)), np.zeros(len(act_n))])

    seed = cfg.get("seed", 42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg["cav"]["test_size"], random_state=seed
    )
    clf = SGDClassifier(
        alpha=cfg["cav"]["alpha"],
        max_iter=cfg["cav"]["max_iter"],
        tol=1e-3,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)

    cav = torch.tensor(clf.coef_[0]).float().to(device)
    return cav, clf.score(X_val, y_val)

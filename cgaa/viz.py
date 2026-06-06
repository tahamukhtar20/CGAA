"""Visualizations of CGAA attack trajectories and feature drift."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def normalize_noise(t: torch.Tensor) -> torch.Tensor:
    """Scale a (N, C, H, W) tensor to [0, 1] per-sample for visualization."""
    t = t.clone()
    t_min = t.view(t.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    t_max = t.view(t.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    t = t - t_min
    return t / (t_max - t_min + 1e-8)


def generate_heatmap(img_t: torch.Tensor, noise_t: torch.Tensor) -> torch.Tensor:
    """Overlay a JET heatmap of noise_t magnitude onto img_t."""
    import cv2

    img = img_t.detach().permute(1, 2, 0).cpu().numpy()
    noise = noise_t.detach().permute(1, 2, 0).cpu().numpy()

    mag = np.linalg.norm(noise, axis=2)
    mag = np.clip(mag, 0, np.percentile(mag, 98))
    mag = mag / (mag.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * mag), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy((img * 0.6) + (heatmap * 0.4)).permute(2, 0, 1)


def plot_manifold_trajectory(tracker, results_dir: str, exp_name: str):
    """Project the high-dim activation path onto 2D PCA."""
    acts = np.vstack(tracker.acts_history)
    acts_2d = PCA(n_components=2).fit_transform(acts)

    plt.figure(figsize=(8, 6))
    plt.plot(acts_2d[:, 0], acts_2d[:, 1], marker="o", markersize=4, alpha=0.6, label="Attack trajectory")
    plt.scatter(acts_2d[0, 0], acts_2d[0, 1], c="green", s=100, label="Start (original)")
    plt.scatter(acts_2d[-1, 0], acts_2d[-1, 1], c="red", s=100, label="End (adversarial)")
    plt.title(f"Manifold Trajectory: {exp_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = os.path.join(results_dir, f"{exp_name}_manifold.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_cosine_similarity(tracker, cav, results_dir: str, exp_name: str):
    """Cosine similarity between per-step activation updates and the CAV."""
    acts = np.vstack(tracker.acts_history)
    acts_diff = acts[1:] - acts[:-1]
    acts_diff_norm = acts_diff / (np.linalg.norm(acts_diff, axis=1, keepdims=True) + 1e-8)

    cav_np = cav.cpu().numpy().reshape(1, -1)
    cav_norm = cav_np / np.linalg.norm(cav_np)
    sims = np.dot(acts_diff_norm, cav_norm.T).flatten()

    plt.figure(figsize=(8, 4))
    plt.plot(sims, color="purple", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title(f"Alignment with Concept (Cosine Sim): {exp_name}")
    plt.xlabel("Attack step")
    plt.ylabel("Cosine similarity")
    plt.ylim(-1.1, 1.1)

    path = os.path.join(results_dir, f"{exp_name}_orthogonality.png")
    plt.savefig(path)
    plt.close()
    return path


def visualize_flow(model, x, cav, layer_name, results_dir: str, exp_name: str,
                   smooth_sigma: float = 3.0, density: float = 1.5):
    """Concept saliency map and ascent-direction streamplot.

    Per-pixel saliency is the channel-wise L2 norm of the input-space concept gradient.
    The streamplot shows the spatial gradient of the (lightly smoothed) saliency map.
    """
    x_in = x.clone().detach().requires_grad_(True)

    layer_mod = dict(model[1].named_modules())[layer_name]
    activations = {}

    def h(_m, _i, o):
        activations["v"] = o.mean(dim=[2, 3])

    handle = layer_mod.register_forward_hook(h)
    try:
        model(x_in)
        concept_score = torch.matmul(activations["v"], cav.to(x.device)).mean()
        grad = torch.autograd.grad(concept_score, x_in)[0]
    finally:
        handle.remove()

    saliency = grad[0].norm(dim=0).cpu().numpy()

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        saliency_smooth = gaussian_filter(saliency, sigma=smooth_sigma)
    else:
        saliency_smooth = saliency

    dy, dx = np.gradient(saliency_smooth)

    H, W = saliency.shape
    Y, X = np.mgrid[0:H, 0:W]

    x_show = x[0].permute(1, 2, 0).detach().cpu().numpy()
    x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(x_show, alpha=0.55)
    ax.imshow(saliency, cmap="hot", alpha=0.35)

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    if magnitude.max() > 0:
        ax.streamplot(X, Y, dx, dy, color=magnitude, cmap="cool", density=density, linewidth=1.0)

    ax.set_axis_off()
    ax.set_title(f"Concept Flow Field: {exp_name}")

    path = os.path.join(results_dir, f"{exp_name}_flow.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _reference_activations(model, loader, layer_name, device, n_samples=100):
    acts = []
    count = 0
    layer_mod = dict(model[1].named_modules())[layer_name]
    tmp = {}

    def h(_m, _i, o):
        tmp["v"] = o.mean(dim=[2, 3])

    handle = layer_mod.register_forward_hook(h)
    try:
        with torch.no_grad():
            for x_batch, _ in loader:
                model(x_batch.to(device))
                acts.append(tmp["v"].cpu().numpy())
                count += x_batch.size(0)
                if count >= n_samples:
                    break
    finally:
        handle.remove()
    return np.vstack(acts)[:n_samples]


def plot_tsne_trajectory(tracker, model, loader, layer_name, device, results_dir: str, exp_name: str):
    """t-SNE of attack trajectory against a cloud of real same-class activations."""
    traj_acts = np.vstack(tracker.acts_history)
    bg_acts = _reference_activations(model, loader, layer_name, device, n_samples=100)

    combined = np.vstack([bg_acts, traj_acts])
    embedded = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(combined)
    bg_2d = embedded[: len(bg_acts)]
    traj_2d = embedded[len(bg_acts):]

    plt.figure(figsize=(10, 8))
    plt.scatter(bg_2d[:, 0], bg_2d[:, 1], c="lightgray", s=50, alpha=0.6, label=f"Real {exp_name} samples")
    plt.plot(traj_2d[:, 0], traj_2d[:, 1], c="black", alpha=0.3, linewidth=1)
    plt.scatter(
        traj_2d[:, 0], traj_2d[:, 1],
        c=np.linspace(0, 1, len(traj_2d)), cmap="turbo", s=60, zorder=10,
    )
    plt.text(traj_2d[0, 0], traj_2d[0, 1], " START", fontsize=12, fontweight="bold")
    plt.text(traj_2d[-1, 0], traj_2d[-1, 1], " END", fontsize=12, fontweight="bold")
    plt.title(f"Manifold Invasion: Path to '{exp_name}'", fontsize=14)
    plt.axis("off")
    plt.legend()

    path = os.path.join(results_dir, f"{exp_name}_tsne.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_layer_drift(model, x_clean, x_adv, results_dir: str, exp_name: str,
                     layers=("layer1", "layer2", "layer3", "layer4")):
    """Relative L2 drift of features at each major ResNet layer."""
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(_m, _i, o):
            activations[name] = o
        return hook

    resnet = model[1]
    for layer in layers:
        hooks.append(dict(resnet.named_modules())[layer].register_forward_hook(make_hook(layer)))

    try:
        with torch.no_grad():
            model(x_clean)
            clean_acts = {k: v.clone() for k, v in activations.items()}
            model(x_adv)
            adv_acts = {k: v.clone() for k, v in activations.items()}
    finally:
        for h in hooks:
            h.remove()

    diffs = [
        (torch.norm(adv_acts[l] - clean_acts[l]) / torch.norm(clean_acts[l])).item()
        for l in layers
    ]

    plt.figure(figsize=(6, 5))
    plt.bar(list(layers), diffs, color="steelblue")
    plt.title("Semantic Perturbation Depth")
    plt.ylabel("Relative feature shift (L2)")
    plt.ylim(0, max(diffs) * 1.2)
    for i, v in enumerate(diffs):
        plt.text(i, v + max(diffs) * 0.01, f"{v:.2f}", ha="center")

    path = os.path.join(results_dir, f"{exp_name}_layer_drift.png")
    plt.savefig(path)
    plt.close()
    return path

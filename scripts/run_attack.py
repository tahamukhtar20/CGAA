"""Run a CGAA experiment: BIM baseline vs. CGAA, plus diagnostic plots.

Usage:
    python scripts/run_attack.py --config configs/experiments/zebra_stripes.yaml
"""
import argparse
import os
import sys

import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgaa.attacks import BIM, CGAA
from cgaa.config import load_config, set_seed
from cgaa.data import get_target_loader
from cgaa.metrics import evaluate
from cgaa.model import get_model
from cgaa.tracker import AttackTracker
from cgaa.viz import (
    normalize_noise,
    plot_cosine_similarity,
    plot_layer_drift,
    plot_manifold_trajectory,
    plot_tsne_trajectory,
    visualize_flow,
)


def _format_metrics(prefix, m):
    return f"{prefix}: " + " | ".join(f"{k}={v:.4f}" for k, v in m.items())


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    exp_name = cfg.get("experiment")
    if exp_name is None or exp_name not in cfg["concept_bank"]:
        raise ValueError(f"Config must set 'experiment' to one of {list(cfg['concept_bank'])}")

    entry = cfg["concept_bank"][exp_name]
    concept, direction = entry["concept"], entry["direction"]
    layer = cfg["concept_layers"][concept]
    results_dir = cfg["paths"]["results_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg, device)

    cav_path = os.path.join(cfg["paths"]["cav_dir"], f"{concept}_{layer}.pt")
    if not os.path.exists(cav_path):
        raise FileNotFoundError(f"CAV not found: {cav_path}. Run scripts/train_concepts.py first.")
    cav = torch.load(cav_path, map_location=device)

    tracker = AttackTracker(cav)
    loader = get_target_loader(cfg, exp_name)

    x_batch, y_batch = next(iter(loader))
    x, y = x_batch.to(device), y_batch.to(device)

    atk = cfg["attack"]

    print("[+] BIM (control)")
    bim = BIM(model, eps=atk["eps"], alpha=atk["alpha"], steps=atk["steps"])
    x_pgd = bim.forward(x, y)
    print(_format_metrics("    BIM ", evaluate(model, x, x_pgd, cav, layer)))

    print(f"[+] CGAA (targeting concept='{concept}', direction={direction:+d})")
    cgaa = CGAA(
        model, cav, layer,
        direction=direction, lambda_val=atk["lambda_val"],
        eps=atk["eps"], alpha=atk["alpha"], steps=atk["steps"],
    )
    x_cgaa = cgaa.forward(x, y, callback=tracker.callback)
    print(_format_metrics("    CGAA", evaluate(model, x, x_cgaa, cav, layer)))

    print("[+] Plots")
    print("    ", plot_manifold_trajectory(tracker, results_dir, exp_name))
    print("    ", plot_cosine_similarity(tracker, cav, results_dir, exp_name))
    print("    ", visualize_flow(model, x[:1], cav, layer, results_dir, exp_name))
    print("    ", plot_tsne_trajectory(tracker, model, loader, layer, device, results_dir, exp_name))
    print("    ", plot_layer_drift(model, x, x_cgaa, results_dir, exp_name))

    vis_x = x[:1]
    vis_pgd = x_pgd[:1]
    vis_cgaa = x_cgaa[:1]
    concept_diff = vis_cgaa - vis_pgd
    blank = torch.ones_like(vis_x)

    row1 = torch.cat([vis_x, vis_pgd, vis_cgaa, normalize_noise(concept_diff)], dim=3)
    row2 = torch.cat(
        [blank, normalize_noise(vis_pgd - vis_x), normalize_noise(vis_cgaa - vis_x), normalize_noise(concept_diff)],
        dim=3,
    )
    grid = torch.cat([row1, row2], dim=2)

    save_path = os.path.join(results_dir, f"{exp_name}.png")
    save_image(grid, save_path)
    print(f"[+] Contact sheet: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML (e.g. configs/experiments/zebra_stripes.yaml)")
    args = parser.parse_args()
    main(args.config)

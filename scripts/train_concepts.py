"""Train Concept Activation Vectors (CAVs) from DTD texture images.

Usage:
    python scripts/train_concepts.py [--config configs/default.yaml]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgaa.cav import train_cav
from cgaa.config import load_config, set_seed
from cgaa.data import get_concept_loaders
from cgaa.model import get_model


def main(config_path: str | None):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg, device)

    for exp_name, entry in cfg["concept_bank"].items():
        concept = entry["concept"]
        layer = cfg["concept_layers"][concept]
        save_path = os.path.join(cfg["paths"]["cav_dir"], f"{concept}_{layer}.pt")

        if os.path.exists(save_path):
            print(f"[=] {concept}/{layer} already trained: {save_path}")
            continue

        print(f"[+] Training CAV: {concept} ({layer})")
        pos, neg = get_concept_loaders(cfg, concept)
        cav, acc = train_cav(model, pos, neg, layer, device, cfg)

        torch.save(cav, save_path)
        print(f"    -> val acc: {acc:.2%} | saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional experiment YAML to overlay.")
    args = parser.parse_args()
    main(args.config)

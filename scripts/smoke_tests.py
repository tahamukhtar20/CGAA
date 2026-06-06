"""Phase 1 visual smoke tests for CGAA.

Answers: does CGAA edit the named concept where the target object appears,
or does it just find an adversarial direction that moves a linear probe's
output?

Runs on CPU. Produces visual grids under ``results/smoke/``. Each grid is
designed for human inspection.

    python scripts/smoke_tests.py [--config configs/default.yaml] [--num-images 5]

Tests:
    1. Eyeball grid           (zebra suppress-stripes; apple inject-stripes)
    2. Perturbation localization heatmap (where does the attack spend its budget)
    3. Direction symmetry     (same zebra, d=+1 vs d=-1)
    4. Cross-concept distinct (same zebra, striped CAV vs dotted CAV)
    5. (Deferred, needs CLIP) Quick CLIP score
    6. (Optional) Masked vs unrestricted
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgaa.attacks import BIM, CGAA
from cgaa.cav import train_cav
from cgaa.config import load_config, set_seed
from cgaa.data import SingleClassDataset, get_concept_loaders
from cgaa.metrics import evaluate
from cgaa.model import get_model
from cgaa.viz import normalize_noise




def _transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def load_target_images(cfg, exp_name: str, n: int):
    entry = cfg["concept_bank"][exp_name]
    root = os.path.join(cfg["paths"]["imagenet_dir"], "train")
    ds = SingleClassDataset(root=root, wnid=entry["wnid"], class_index=entry["target_id"], transform=_transforms())
    n = min(n, len(ds))
    xs, ys = zip(*[ds[i] for i in range(n)])
    return torch.stack(xs), torch.tensor(ys)


def ensure_cav(cfg, model, concept: str, layer: str, device):
    path = os.path.join(cfg["paths"]["cav_dir"], f"{concept}_{layer}.pt")
    if os.path.exists(path):
        print(f"[=] CAV cached: {path}")
        return torch.load(path, map_location=device)
    print(f"[+] Training CAV: {concept}/{layer}")
    pos, neg = get_concept_loaders(cfg, concept)
    cav, acc = train_cav(model, pos, neg, layer, device, cfg)
    torch.save(cav, path)
    print(f"    val acc: {acc:.2%} | saved: {path}")
    return cav


def run_bim(model, x, y, cfg):
    a = cfg["attack"]
    bim = BIM(model, eps=a["eps"], alpha=a["alpha"], steps=a["steps"])
    print(f"    BIM ({a['steps']} steps) on {x.size(0)} images")
    return bim.forward(x, y)


def run_cgaa(model, x, y, cav, layer, direction, cfg, tag):
    a = cfg["attack"]
    atk = CGAA(
        model, cav, layer,
        direction=direction, lambda_val=a["lambda_val"],
        eps=a["eps"], alpha=a["alpha"], steps=a["steps"],
    )
    print(f"    CGAA {tag} (d={direction:+d}, λ={a['lambda_val']}) on {x.size(0)} images")
    return atk.forward(x, y)


def per_pixel_magnitude(delta: torch.Tensor) -> torch.Tensor:
    """Mean absolute magnitude over channels, per (N, H, W)."""
    return delta.abs().mean(dim=1, keepdim=True)  # (N, 1, H, W)


def colorize_heatmap(heat: torch.Tensor, cmap: str = "hot") -> torch.Tensor:
    """heat: (N, 1, H, W) in [0, 1]. Returns (N, 3, H, W)."""
    cm = plt.get_cmap(cmap)
    h = heat.squeeze(1).cpu().numpy()  # (N, H, W)
    h = h / (h.reshape(h.shape[0], -1).max(axis=1)[:, None, None] + 1e-8)
    rgb = cm(h)[..., :3]  # (N, H, W, 3)
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).float()


def blend(img: torch.Tensor, heat_rgb: torch.Tensor, alpha: float = 0.55) -> torch.Tensor:
    return (1 - alpha) * img + alpha * heat_rgb


def save_grid(tiles_per_row, col_labels, out_path: Path, title: str):
    """Save a clean image grid without any overlaid text. Captions are in the README."""
    cols = torch.stack(tiles_per_row, dim=1)
    n, k = cols.shape[:2]
    flat = cols.reshape(n * k, *cols.shape[2:])
    grid = make_grid(flat.clamp(0, 1), nrow=k, padding=6, pad_value=1.0).cpu().permute(1, 2, 0).numpy()

    H, W = grid.shape[:2]
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.imshow(grid)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)




def test1_eyeball(x, x_bim, x_cgaa, out: Path, title: str):
    cols = [
        x,
        x_bim,
        x_cgaa,
        normalize_noise(x_cgaa - x),
        normalize_noise(x_cgaa - x_bim),
    ]
    save_grid(cols, ["Original", "BIM", "CGAA", "Δ CGAA vs orig", "Δ CGAA vs BIM"], out, title)


def test2_localization(x, x_bim, x_cgaa, out: Path, title: str):
    bim_heat = colorize_heatmap(per_pixel_magnitude(x_bim - x))
    cg_heat = colorize_heatmap(per_pixel_magnitude(x_cgaa - x))
    cols = [
        x,
        blend(x, bim_heat),
        blend(x, cg_heat),
        bim_heat,
        cg_heat,
    ]
    save_grid(cols, ["Original", "BIM overlay", "CGAA overlay", "BIM |δ|", "CGAA |δ|"], out, title)


def test3_direction(x, x_cgaa_pos, x_cgaa_neg, out: Path, title: str):
    cols = [
        x,
        x_cgaa_pos,
        x_cgaa_neg,
        normalize_noise(x_cgaa_pos - x),
        normalize_noise(x_cgaa_neg - x),
        normalize_noise(x_cgaa_pos - x_cgaa_neg),
    ]
    save_grid(
        cols,
        ["Original", "CGAA d=+1", "CGAA d=-1", "Δ(+1−orig)", "Δ(−1−orig)", "Δ(+1 vs −1)"],
        out,
        title,
    )


def test4_cross_concept(x, x_striped, x_dotted, out: Path, title: str):
    cols = [
        x,
        x_striped,
        x_dotted,
        normalize_noise(x_striped - x),
        normalize_noise(x_dotted - x),
        normalize_noise(x_striped - x_dotted),
    ]
    save_grid(
        cols,
        ["Original", "CGAA striped", "CGAA dotted", "Δ striped", "Δ dotted", "Δ striped vs dotted"],
        out,
        title,
    )




def metrics_summary(model, x, x_adv, cav, layer, tag: str):
    m = evaluate(model, x, x_adv, cav, layer)
    print(f"    {tag}:  ASR={m['ASR']:.2f}  Δ(train_cav)={m['Delta']:+.3f}  L2={m['L2']:.3f}")
    return m




def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="Optional experiment YAML to overlay on default.")
    p.add_argument("--num-images", type=int, default=5, help="Images per target class (default: 5).")
    p.add_argument("--steps", type=int, default=None, help="Override attack steps (smoke: try 25 for speed).")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    if args.steps is not None:
        cfg["attack"]["steps"] = args.steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}  steps: {cfg['attack']['steps']}  num_images: {args.num_images}")

    out = Path(cfg["paths"]["results_dir"]) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[+] Output: {out}")

    print("[+] Loading model...")
    model = get_model(cfg, device)
    layer = cfg["concept_layers"]["striped"]

    print("[+] Ensuring CAVs...")
    striped_cav = ensure_cav(cfg, model, "striped", layer, device)
    dotted_cav = ensure_cav(cfg, model, "dotted", layer, device)

    print("[+] Loading target images...")
    zx, zy = load_target_images(cfg, "zebra_stripes", args.num_images)
    ax, ay = load_target_images(cfg, "apple_stripes", args.num_images)
    zx, zy, ax, ay = zx.to(device), zy.to(device), ax.to(device), ay.to(device)

    print("[+] Running attacks...")
    print("  zebra:")
    z_bim = run_bim(model, zx, zy, cfg)
    z_cgaa_sup = run_cgaa(model, zx, zy, striped_cav, layer, -1, cfg, "striped-suppress")
    z_cgaa_inj = run_cgaa(model, zx, zy, striped_cav, layer, +1, cfg, "striped-inject")
    z_cgaa_dot = run_cgaa(model, zx, zy, dotted_cav, layer, +1, cfg, "dotted-inject")
    print("  apple:")
    a_bim = run_bim(model, ax, ay, cfg)
    a_cgaa = run_cgaa(model, ax, ay, striped_cav, layer, +1, cfg, "striped-inject")

    print("[+] Metrics (measured against training CAV — circular, for sanity only):")
    metrics_summary(model, zx, z_bim, striped_cav, layer, "zebra   BIM       ")
    metrics_summary(model, zx, z_cgaa_sup, striped_cav, layer, "zebra   CGAA(-1) ")
    metrics_summary(model, zx, z_cgaa_inj, striped_cav, layer, "zebra   CGAA(+1) ")
    metrics_summary(model, zx, z_cgaa_dot, dotted_cav, layer, "zebra   CGAA dotted")
    metrics_summary(model, ax, a_bim, striped_cav, layer, "apple   BIM       ")
    metrics_summary(model, ax, a_cgaa, striped_cav, layer, "apple   CGAA(+1) ")

    print("[+] Building grids...")
    test1_eyeball(zx, z_bim, z_cgaa_sup, out / "test1_zebra_suppress.png",
                  "Test 1: Zebra — suppress stripes (CGAA d=−1)")
    test1_eyeball(ax, a_bim, a_cgaa, out / "test1_apple_inject.png",
                  "Test 1: Apple — inject stripes (CGAA d=+1)")
    test2_localization(zx, z_bim, z_cgaa_sup, out / "test2_zebra_localization.png",
                       "Test 2: Zebra — where does the attack spend its budget?")
    test2_localization(ax, a_bim, a_cgaa, out / "test2_apple_localization.png",
                       "Test 2: Apple — where does the attack spend its budget?")
    test3_direction(zx, z_cgaa_inj, z_cgaa_sup, out / "test3_zebra_direction.png",
                    "Test 3: Zebra — direction symmetry (+1 vs −1, same CAV)")
    test4_cross_concept(zx, z_cgaa_inj, z_cgaa_dot, out / "test4_zebra_cross_concept.png",
                        "Test 4: Zebra — cross-concept distinctness (striped CAV vs dotted CAV, both d=+1)")

    print(f"\n[+] Done. Grids written to {out}/")
    print("    Inspect with fresh eyes. Look for:")
    print("    - Test 1: are stripes actually fading/appearing on the target object, or is it speckle noise?")
    print("    - Test 2: does CGAA's heatmap concentrate on the animal/fruit, or on background?")
    print("    - Test 3: are d=+1 and d=-1 visually DIFFERENT, in a concept-consistent way?")
    print("    - Test 4: do striped and dotted perturbations look distinct, or interchangeable?")


if __name__ == "__main__":
    main()

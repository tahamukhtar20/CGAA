# Setup and Usage

## Requirements

Python 3.10+. ImageNet (ILSVRC 2012) training split is required for running attacks.
DTD (Describable Textures Dataset) images for the `banded` and `striped` categories
are included in this repository under `dtd/images/`.

```
pip install torch torchvision scikit-learn matplotlib scipy pyyaml
```

`opencv-python` is required only for `viz.generate_heatmap`. It is not needed for
running attacks or smoke tests.

## Data paths

Edit `configs/default.yaml` to point `paths.imagenet_dir` at your ImageNet root.
The directory should contain `train/` with WNID subfolders (standard ImageNet layout).

## Step 1: Train CAVs

```
python scripts/train_concepts.py
```

Trains linear classifiers on DTD activations at layer4 for the `striped` and `dotted`
concepts. Saves to `results/cavs/`. Idempotent: skips already-trained CAVs.

## Step 2: Run smoke tests

```
python scripts/smoke_tests.py --num-images 5 --steps 50
```

Runs the four visual tests on CPU. Writes grids to `results/smoke/`. Expected runtime:
5-10 minutes on CPU for 5 images.

## Step 3: Run a full experiment

```
python scripts/run_attack.py --config configs/experiments/zebra_stripes.yaml
```

Runs BIM vs CGAA on the zebra class with all diagnostic plots (manifold trajectory,
cosine similarity, flow field, t-SNE, layer drift). Writes to `results/`.

## Configuration

All hyperparameters live in `configs/default.yaml`. Experiment YAMLs under
`configs/experiments/` override specific keys. The concept bank defines which
ImageNet class and DTD concept to use per experiment.

Key parameters:
- `attack.eps`: L-infinity bound (default 16/255)
- `attack.lambda_val`: concept-loss weight (default 100.0)
- `attack.steps`: BIM iterations (default 50)
- `cav.samples_per_class`: DTD images per class for CAV training (default 100)

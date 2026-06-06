import os
import random
from typing import Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _default_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


class SingleClassDataset(Dataset):
    """All images under ``root/wnid/`` as (tensor, class_index) pairs."""

    def __init__(self, root: str, wnid: str, class_index: int, transform=None):
        self.root = os.path.join(root, wnid)
        self.transform = transform
        self.class_index = class_index

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Folder {self.root} not found.")

        self.samples = [
            os.path.join(self.root, f)
            for f in os.listdir(self.root)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_index


def get_target_loader(cfg, experiment: str) -> DataLoader:
    """DataLoader over the ImageNet-train folder for the experiment's target class."""
    entry = cfg["concept_bank"][experiment]
    root = os.path.join(cfg["paths"]["imagenet_dir"], "train")
    ds = SingleClassDataset(
        root=root,
        wnid=entry["wnid"],
        class_index=entry["target_id"],
        transform=_default_transforms(),
    )
    return DataLoader(ds, batch_size=cfg["attack"]["batch_size"], shuffle=False, num_workers=4)


def get_concept_loaders(cfg, concept_name: str) -> Tuple[DataLoader, DataLoader]:
    """Positive/negative DataLoaders for training a CAV.

    Positives = images in the DTD folder ``<concept_name>/``.
    Negatives = a random sample of images from all other DTD folders.
    """
    dtd_path = os.path.join(cfg["paths"]["dtd_dir"], "images")
    if not os.path.isdir(dtd_path):
        dtd_path = cfg["paths"]["dtd_dir"]

    ds = datasets.ImageFolder(dtd_path, transform=_default_transforms())
    concept_idx = ds.class_to_idx.get(concept_name)
    if concept_idx is None:
        raise KeyError(f"Concept '{concept_name}' not found in {dtd_path}")

    n = cfg["cav"]["samples_per_class"]
    batch_size = cfg["cav"]["batch_size"]

    pos = [i for i, (_, y) in enumerate(ds.samples) if y == concept_idx][:n]
    neg_pool = list(set(range(len(ds))) - set(pos))
    neg = random.sample(neg_pool, min(n, len(neg_pool)))

    l_pos = DataLoader(Subset(ds, pos), batch_size=batch_size, shuffle=False)
    l_neg = DataLoader(Subset(ds, neg), batch_size=batch_size, shuffle=False)
    return l_pos, l_neg

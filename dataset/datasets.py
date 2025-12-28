import torch
from torchvision import datasets, transforms

def get_dataloaders(args, tf):
    ds = datasets.ImageFolder(args.data_dir, transform=tf)
    valid_idx = [i for i, c in enumerate(ds.classes) if c in ['random', args.concept]]
    ds.samples = [s for s in ds.samples if s[1] in valid_idx]
    ds.targets = [s[1] for s in ds.samples]
    concept_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=8, shuffle=False)

    test_ds = datasets.CIFAR10(args.cifar_dir, train=False, download=True, transform=tf)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    return concept_loader, test_loader, ds.targets

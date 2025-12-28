import torch
from torchvision import transforms, utils
from torchattacks import PGD
import os, numpy as np
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import argparse

from util.utils import set_seed, get_phi
from model.models import get_model
from dataset.datasets import get_dataloaders
from model.models import CGAA

set_seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     type=str,   default="dtd/images")
    parser.add_argument("--concept",      type=str,   default="striped")  # Dynamic concept
    parser.add_argument("--target_layer", type=str,   default="model.layer3") # Configurable layer
    parser.add_argument("--cifar_dir",    type=str,   default="data")
    parser.add_argument("--eps",          type=float, default=8/255)
    parser.add_argument("--alpha",        type=float, default=2/255)
    parser.add_argument("--steps",        type=int,   default=10)
    parser.add_argument("--lam",          type=float, default=2.0)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--limit",        type=int,   default=512)
    parser.add_argument("--vis_limit",    type=int,   default=5)
    return parser.parse_args()

def run():
    args = get_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mu = torch.tensor([0.4914, 0.4822, 0.4465]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).to(device).view(1, 3, 1, 1)
    
    model = get_model(device, mu, std)
    tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    
    concept_loader, test_loader, concept_labels = get_dataloaders(args, tf)

    print("[+] Training CAV (v_C)")
    phi_acts = get_phi(model, concept_loader, device, layer_name=args.target_layer)    
    clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42).fit(phi_acts.numpy(), np.array(concept_labels))
    v_C = torch.from_numpy(clf.coef_[0]).float()

    pgd = PGD(model, eps=args.eps, alpha=args.alpha, steps=args.steps)
    cgaa = CGAA(model, v_C, args.lam, device, target_layer=args.target_layer, eps=args.eps, alpha=args.alpha, steps=args.steps)

    acc_pgd, acc_cgaa, total, vis_count = 0, 0, 0, 0
    for x, y in tqdm(test_loader, desc="Testing"):
        x, y = x.to(device), y.to(device)
        x_pgd, x_cgaa = pgd(x, y), cgaa(x, y)

        if vis_count < args.vis_limit:
            for i in range(min(x.size(0), args.vis_limit - vis_count)):
                path = f"qualitative/sample_{vis_count + i}"
                os.makedirs(path, exist_ok=True)
                utils.save_image(x[i], f"{path}/image.png")
                utils.save_image(x_pgd[i], f"{path}/pgd.png")
                utils.save_image(x_cgaa[i], f"{path}/cgaa.png")
            vis_count += x.size(0)

        with torch.no_grad():
            acc_pgd += (model(x_pgd).argmax(1) == y).sum().item()
            acc_cgaa += (model(x_cgaa).argmax(1) == y).sum().item()
        
        total += y.size(0)
        if total >= args.limit: break

    print(f"\n[!] RESULTS\n[#] Samples: {total}\n[#] PGD Acc: {acc_pgd/total:.2%}\n[#] CGAA Acc: {acc_cgaa/total:.2%}")

if __name__ == "__main__":
    run()
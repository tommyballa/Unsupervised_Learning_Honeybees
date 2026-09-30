"""
Diagnostica FastFlow su Varroa: capire se il problema e' il modello o i dati.

Uso:
    python experiments/diagnose_fastflow_varroa.py --checkpoint ./checkpoints/fastflow_single_bee.pth

Stampa:
  A. statistiche del test set (quota anomalie, area delle maschere, file mancanti)
  B. distribuzione degli score su train / test normali / test anomali
     + AUROC train vs test-normali (domain shift)
  C. localizzazione: pixel AUROC e quante volte la mappa e' piu' alta dentro la maschera
  D. AUROC con score alternativi (max, mean, top-1%) calcolati dalle stesse mappe
  E. quota di massimi che cadono sul bordo dell'immagine
  F. stesso AUROC con un flow NON addestrato (baseline)
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.fastflow.fastflow import create_fastflow
from moviad.utilities.configurations import Split


def build_args(train_dir, img_size):
    return DatasetArguments(
        dataset_path=train_dir,
        img_size=(img_size, img_size),
        gt_mask_size=(img_size, img_size),
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )


@torch.no_grad()
def run_model(model, loader, device):
    model.eval()
    maps, masks, labels = [], [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            image, label, mask, _ = batch
        else:
            image, label, mask = batch, torch.zeros(batch.shape[0]), None
        anom_maps, _ = model(image.to(device).float())
        maps.append(anom_maps.float().cpu().numpy()[:, 0])
        labels.append(np.asarray(label))
        if mask is not None:
            masks.append(mask.cpu().numpy()[:, 0] > 0)
    maps = np.concatenate(maps)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks) if masks else None
    return maps, masks, labels


def image_scores(maps):
    flat = maps.reshape(maps.shape[0], -1)
    k = max(1, flat.shape[1] // 100)
    return {
        "max": flat.max(1),
        "mean": flat.mean(1),
        "top1%": np.sort(flat, axis=1)[:, -k:].mean(1),
    }


def describe(name, x):
    print(f"  {name:<14} n={len(x):5d}  mean={x.mean():.4g}  std={x.std():.4g}  "
          f"p5={np.percentile(x, 5):.4g}  p95={np.percentile(x, 95):.4g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_dir", default="/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees")
    p.add_argument("--backbone", default="wide_resnet50_2")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--n_train", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    a = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = build_args(a.train_dir, a.img_size)
    test_ds = VarroaDataset(args, "varroa", Split.TEST)
    train_ds = VarroaDataset(args, "varroa", Split.TRAIN)

    # ---------------- A. test set ----------------
    print("\n=== A. Test set ===")
    labels = np.asarray(test_ds.labels)
    print(f"  immagini: {len(labels)}  anomale: {labels.sum()} ({labels.mean():.2%})")
    missing = [pth for pth in test_ds.image_paths + [m for m in test_ds.mask_paths if m] if not os.path.exists(pth)]
    print(f"  file mancanti: {len(missing)}" + (f"  es. {missing[:3]}" if missing else ""))
    areas, sizes = [], set()
    for img_p, m_p in zip(test_ds.image_paths, test_ds.mask_paths):
        if m_p and os.path.exists(m_p):
            m = np.array(Image.open(m_p).convert("L")) > 127
            areas.append(m.mean())
            sizes.add(Image.open(img_p).size)
    if areas:
        areas = np.array(areas)
        px = areas * a.img_size ** 2
        print(f"  area maschera (frazione immagine): mean={areas.mean():.4%} min={areas.min():.4%} max={areas.max():.4%}")
        print(f"  => a {a.img_size}px la varroa copre in media ~{px.mean():.0f} px (~{np.sqrt(px.mean()):.1f}x{np.sqrt(px.mean()):.1f})")
        print(f"  maschere vuote: {(areas == 0).sum()}")
        print(f"  dimensioni originali immagini anomale (prime 5): {list(sizes)[:5]}")

    # ---------------- modello addestrato ----------------
    model = create_fastflow((a.img_size, a.img_size), a.backbone, device)
    state = torch.load(a.checkpoint, map_location=device)
    missing_k, unexpected_k = model.load_state_dict(state, strict=False)
    print(f"\n  checkpoint caricato: missing={len(missing_k)} unexpected={len(unexpected_k)}")
    if missing_k or unexpected_k:
        print(f"  ATTENZIONE: chiavi non corrispondenti, es. {(missing_k + unexpected_k)[:5]}")
    model.to(device)

    test_loader = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False, num_workers=4)
    idx = np.random.RandomState(0).choice(len(train_ds), min(a.n_train, len(train_ds)), replace=False)
    train_loader = DataLoader(Subset(train_ds, idx), batch_size=a.batch_size, shuffle=False, num_workers=4)

    t_maps, t_masks, t_labels = run_model(model, test_loader, device)
    tr_maps, _, _ = run_model(model, train_loader, device)

    # ---------------- B. distribuzioni ----------------
    print("\n=== B. Score immagine (max della mappa) ===")
    s_test = image_scores(t_maps)["max"]
    s_train = image_scores(tr_maps)["max"]
    describe("train", s_train)
    describe("test normali", s_test[t_labels == 0])
    describe("test anomali", s_test[t_labels == 1])
    y = np.r_[np.zeros(len(s_train)), np.ones((t_labels == 0).sum())]
    auc_shift = roc_auc_score(y, np.r_[s_train, s_test[t_labels == 0]])
    print(f"  AUROC train vs test-normali: {auc_shift:.3f}  (~0.5 = stessa distribuzione, >>0.5 = domain shift)")

    # ---------------- C. localizzazione ----------------
    print("\n=== C. Localizzazione sulle immagini anomale ===")
    an = t_labels == 1
    m_an, g_an = t_maps[an], t_masks[an]
    valid = g_an.reshape(len(g_an), -1).any(1)
    if valid.any():
        pix_auc = roc_auc_score(g_an[valid].ravel()[::7], m_an[valid].ravel()[::7])
        inside = np.array([mp[g].mean() for mp, g in zip(m_an[valid], g_an[valid])])
        outside = np.array([mp[~g].mean() for mp, g in zip(m_an[valid], g_an[valid])])
        argmax_in = np.array([g.ravel()[mp.argmax()] for mp, g in zip(m_an[valid], g_an[valid])])
        print(f"  pixel AUROC (solo immagini anomale): {pix_auc:.3f}")
        print(f"  mappa piu' alta dentro la maschera che fuori: {(inside > outside).mean():.1%} delle immagini")
        print(f"  argmax della mappa dentro la maschera:        {argmax_in.mean():.1%} delle immagini")

    # ---------------- D. score alternativi ----------------
    print("\n=== D. AUROC immagine con score diversi ===")
    for name, s in image_scores(t_maps).items():
        print(f"  {name:<6} AUROC={roc_auc_score(t_labels, s):.3f}")

    # ---------------- E. bordi ----------------
    print("\n=== E. Posizione del massimo ===")
    H, W = t_maps.shape[1:]
    b = a.img_size // 16
    ys, xs = np.unravel_index(t_maps.reshape(len(t_maps), -1).argmax(1), (H, W))
    border = (ys < b) | (ys >= H - b) | (xs < b) | (xs >= W - b)
    print(f"  massimi entro {b}px dal bordo: {border.mean():.1%} (area bordo = {1 - (1 - 2 * b / H) * (1 - 2 * b / W):.1%})")

    # ---------------- F. baseline non addestrata ----------------
    print("\n=== F. Flow NON addestrato (stesso backbone) ===")
    torch.manual_seed(0)
    untrained = create_fastflow((a.img_size, a.img_size), a.backbone, device)
    u_maps, _, _ = run_model(untrained, test_loader, device)
    for name, s in image_scores(u_maps).items():
        print(f"  {name:<6} AUROC={roc_auc_score(t_labels, s):.3f}")


if __name__ == "__main__":
    main()

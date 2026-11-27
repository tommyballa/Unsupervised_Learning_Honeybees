import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "experiments/single_bee")
import sweep_fastflow as sf, main_fastflow as mf
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.utilities.configurations import Split
ckpt, out = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
sys.argv = ["x", "--tag", "viz"] + extra
args = sf.parse_args(); dev = torch.device("cuda:0")
mf.IMG_SIZE = (args.img_size, args.img_size)
ds = VarroaDataset(mf.dataset_args(), "varroa", Split.TEST)
model = sf.build_model(args, dev); model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()
idx = [i for i in range(len(ds)) if ds.labels[i] == 1][:4] if hasattr(ds, "labels") else None
if idx is None:
    idx = [i for i in range(len(ds)) if ds[i][1] == 1][:4]
idx += [0, 1]
fig, ax = plt.subplots(len(idx), 5, figsize=(15, 3 * len(idx)))
for r, i in enumerate(idx):
    x, y, m, p = ds[i]
    with torch.no_grad():
        z, _ = model.fast_flow_module(sf.features(model, x[None].to(dev)))
    rgb = (x.permute(1, 2, 0).numpy() * mf.IMAGENET_STD + mf.IMAGENET_MEAN).clip(0, 1)
    ax[r, 0].imshow(rgb); ax[r, 1].imshow(m[0], cmap="gray")
    for s in range(3):
        mp = (z[s] ** 2).mean(1)[0].float().cpu().numpy()
        im = ax[r, 2 + s].imshow(mp, cmap="jet"); plt.colorbar(im, ax=ax[r, 2 + s], fraction=0.046)
    for a in ax[r]: a.axis("off")
fig.tight_layout(); fig.savefig(out, dpi=70)

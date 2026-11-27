"""
Orientamento testa in alto delle api normalizzate da dataset_cleaning.py.

La normalizzazione mette l'asse maggiore dell'ape in verticale, ma la testa può
finire sia in alto sia in basso. Qui un classificatore lineare su embedding
DINOv2 decide da che parte sta la testa.

Il classificatore guarda la differenza tra l'embedding dell'ape e quello della
stessa ape ruotata di 180°, quindi lo score è antisimmetrico per costruzione:
ruotare l'ape ne cambia solo il segno.
- score > 0: testa in alto, score < 0: testa in basso.
- Raddrizzare un'ape già raddrizzata non la cambia: il passo è idempotente.

Addestramento (etichette in config/head_orientation_labels.json, pesi in weights/):
    python bee_orientation.py train
Su 214 api di batch3+4+5+6_SAM3 etichettate a mano la cross-validation a 10 fold
non sbaglia nessuna ape; le api ambigue (viste di lato, parziali) non sono tra le
etichettate e lì qualche errore è da aspettarsi.
"""
import argparse
import hashlib
import json
import os

from bee_paths import DATA_ROOT

import cv2
import numpy as np

BACKBONE = "dinov2_vits14"
IMAGE_SIZE = 224
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(REPO_DIR, "weights", "head_orientation_dinov2_vits14.npz")
LABELS_PATH = os.path.join(REPO_DIR, "config", "head_orientation_labels.json")
PROBE_C = 0.1   # Regolarizzazione della regressione logistica (0.03-0.3 danno lo stesso risultato)


def weights_id(weights_path=WEIGHTS_PATH):
    """Impronta dei pesi: se cambia, gli orientamenti calcolati prima vanno ricontrollati."""
    with open(weights_path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


class HeadOrienter:
    """Score testa-in-alto per api già normalizzate (TARGET_SIZE x TARGET_SIZE, BGR)."""

    def __init__(self, weights_path=WEIGHTS_PATH, device="auto", batch_size=64, need_weights=True):
        import torch
        from bee_al.embeddings import IMAGENET_MEAN, IMAGENET_STD, load_backbone, pick_device

        self.torch = torch
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.model = load_backbone(BACKBONE).to(self.device).eval()
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)
        self.w = None
        if need_weights:
            if not os.path.exists(weights_path):
                raise SystemExit(f"Pesi dell'orientamento non trovati in '{weights_path}': "
                                 f"lancia prima 'python bee_orientation.py train'.")
            self.w = np.load(weights_path)["w"]
            self.weights_id = weights_id(weights_path)

    def _prep(self, img_bgr):
        x = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        x = (x[:, :, ::-1].astype(np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)

    def features(self, images):
        """Differenza tra embedding dell'ape e dell'ape ruotata di 180° (N x D)."""
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                chunk = images[i:i + self.batch_size]
                batch = np.stack([self._prep(im) for im in chunk] +
                                 [self._prep(cv2.rotate(im, cv2.ROTATE_180)) for im in chunk])
                emb = self.model(self.torch.from_numpy(batch).to(self.device)).float().cpu().numpy()
                out.append(emb[:len(chunk)] - emb[len(chunk):])
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def scores(self, images):
        """Score per ape: > 0 testa in alto, < 0 testa in basso."""
        if not len(images):
            return np.zeros(0)
        return self.features(images) @ self.w


# ============================================================
# ADDESTRAMENTO
# ============================================================
# Le etichette non dicono "testa in alto nel frame finale", che dipende da come
# dataset_cleaning orienta l'ape, ma la direzione della testa nel crop grezzo
# (vettore unitario x, y). Così restano valide anche se cambia la normalizzazione.

def head_up_in_frame(matrix, head_dir):
    """True se la direzione head_dir del crop grezzo punta verso l'alto nel frame finale."""
    return float((matrix[:, :2] @ np.asarray(head_dir, dtype=float))[1]) < 0


def train(labels_path=LABELS_PATH, weights_path=WEIGHTS_PATH, device="auto"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    import dataset_cleaning as dc

    with open(labels_path) as f:
        labels = json.load(f)["bees"]

    images, y = [], []
    for item in labels:
        # raw_path is stored relative to the dataset root (BEES_DATA_ROOT)
        img, mask, is_raw = dc.load_image_and_mask(os.path.join(DATA_ROOT, item["raw_path"]))
        if img is None:
            print(f"  ✗ Crop non leggibile, salto: {item['raw_path']}")
            continue
        result = dc.clean_bee(img, mask, is_raw)
        if result.image is None:
            print(f"  ✗ Crop scartato dalla pulizia attuale ({result.reason}), salto: {item['raw_path']}")
            continue
        images.append(result.image)
        y.append(int(head_up_in_frame(result.matrix, item["head_dir"])))
    y = np.array(y)
    print(f"Api etichettate usabili: {len(y)} ({y.sum()} con la testa in alto nel frame non orientato)")

    X = HeadOrienter(device=device, need_weights=False).features(images)

    def fit(Xtr, ytr):
        # Ogni ape entra anche capovolta (feature negate, etichetta invertita): senza intercetta
        # il classificatore resta antisimmetrico e non impara una preferenza per il su o il giù.
        clf = LogisticRegression(C=PROBE_C, max_iter=5000, fit_intercept=False)
        return clf.fit(np.vstack([Xtr, -Xtr]), np.r_[ytr, 1 - ytr])

    correct = 0
    for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
        correct += int((fit(X[tr], y[tr]).predict(X[te]) == y[te]).sum())
    cv_acc = correct / len(y)
    print(f"Accuratezza in cross-validation (10 fold): {cv_acc:.3f}")

    w = fit(X, y).coef_[0].astype(np.float32)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    np.savez(weights_path, w=w, backbone=BACKBONE, image_size=IMAGE_SIZE, n_labels=len(y), cv_accuracy=cv_acc)
    print(f"Pesi salvati in '{weights_path}' (id {weights_id(weights_path)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train", help="Addestra il classificatore testa su/giù")
    p_train.add_argument("--labels", default=LABELS_PATH)
    p_train.add_argument("--weights", default=WEIGHTS_PATH)
    p_train.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(args.labels, args.weights, args.device)

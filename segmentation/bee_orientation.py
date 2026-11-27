"""
Head-up orientation of the bees normalised by dataset_cleaning.py.

The normalisation puts the major axis of the bee vertical, but the head can
end up either at the top or at the bottom. Here a linear classifier on DINOv2
embeddings decides which side the head is on.

The classifier looks at the difference between the embedding of the bee and that
of the same bee rotated by 180°, so the score is antisymmetric by construction:
rotating the bee only changes its sign.
- score > 0: head at the top, score < 0: head at the bottom.
- Straightening an already straightened bee does not change it: the step is idempotent.

Training (labels in config/head_orientation_labels.json, weights in weights/):
    python bee_orientation.py train
On 214 hand-labelled bees from batch3+4+5+6_SAM3, 10-fold cross-validation
gets every bee right; ambiguous bees (side views, partial) are not among the
labelled ones, and some errors are to be expected there.
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
PROBE_C = 0.1   # Logistic regression regularisation (0.03-0.3 give the same result)


def weights_id(weights_path=WEIGHTS_PATH):
    """Fingerprint of the weights: if it changes, orientations computed earlier must be re-checked."""
    with open(weights_path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


class HeadOrienter:
    """Head-up score for already normalised bees (TARGET_SIZE x TARGET_SIZE, BGR)."""

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
                raise SystemExit(f"Orientation weights not found in '{weights_path}': "
                                 f"run 'python bee_orientation.py train' first.")
            self.w = np.load(weights_path)["w"]
            self.weights_id = weights_id(weights_path)

    def _prep(self, img_bgr):
        x = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        x = (x[:, :, ::-1].astype(np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)

    def features(self, images):
        """Difference between the embedding of the bee and of the bee rotated by 180° (N x D)."""
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
        """Score per bee: > 0 head at the top, < 0 head at the bottom."""
        if not len(images):
            return np.zeros(0)
        return self.features(images) @ self.w


# ============================================================
# TRAINING
# ============================================================
# The labels do not say "head at the top in the final frame", which depends on how
# dataset_cleaning orients the bee, but the head direction in the raw crop
# (unit vector x, y). This way they stay valid even if the normalisation changes.

def head_up_in_frame(matrix, head_dir):
    """True if the head_dir direction of the raw crop points upwards in the final frame."""
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
            print(f"  ✗ Unreadable crop, skipping: {item['raw_path']}")
            continue
        result = dc.clean_bee(img, mask, is_raw)
        if result.image is None:
            print(f"  ✗ Crop rejected by the current cleaning ({result.reason}), skipping: {item['raw_path']}")
            continue
        images.append(result.image)
        y.append(int(head_up_in_frame(result.matrix, item["head_dir"])))
    y = np.array(y)
    print(f"Usable labelled bees: {len(y)} ({y.sum()} with the head at the top in the unoriented frame)")

    X = HeadOrienter(device=device, need_weights=False).features(images)

    def fit(Xtr, ytr):
        # Each bee also enters upside down (negated features, inverted label): without an intercept
        # the classifier stays antisymmetric and does not learn a preference for up or down.
        clf = LogisticRegression(C=PROBE_C, max_iter=5000, fit_intercept=False)
        return clf.fit(np.vstack([Xtr, -Xtr]), np.r_[ytr, 1 - ytr])

    correct = 0
    for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
        correct += int((fit(X[tr], y[tr]).predict(X[te]) == y[te]).sum())
    cv_acc = correct / len(y)
    print(f"Cross-validation accuracy (10 folds): {cv_acc:.3f}")

    w = fit(X, y).coef_[0].astype(np.float32)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    np.savez(weights_path, w=w, backbone=BACKBONE, image_size=IMAGE_SIZE, n_labels=len(y), cv_accuracy=cv_acc)
    print(f"Weights saved to '{weights_path}' (id {weights_id(weights_path)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train", help="Train the head up/down classifier")
    p_train.add_argument("--labels", default=LABELS_PATH)
    p_train.add_argument("--weights", default=WEIGHTS_PATH)
    p_train.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(args.labels, args.weights, args.device)

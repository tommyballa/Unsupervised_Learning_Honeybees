from bee_paths import DATA_ROOT
import os
import csv
import cv2
import glob
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from collections import defaultdict

PCA_COMPONENTS = 50

# Path to the results directory produced by the active_learning_tool web app
ACTIVE_LEARNING_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "active_learning_tool", "results"
)


def _clean_label(label_str):
    """Normalise a label string exactly as the web tool does."""
    return " ".join(label_str.strip().split()).title()


def _parse_shifted_row_single(row):
    """Handle the column-shift quirk in single_labels.csv (same logic as app.py)."""
    img = str(row.get("image_name", ""))
    lbl = str(row.get("label", ""))
    if "@" in img or img == "anonymous":
        img = lbl
        lbl = str(row.get("confidence", ""))
    return img, lbl


def _parse_shifted_row_grid(row):
    """Handle the column-shift quirk in grid_batch_labels.csv (same logic as app.py)."""
    sel_imgs = str(row.get("selected_images", ""))
    assigned = str(row.get("assigned_label", ""))
    if "@" in sel_imgs or sel_imgs == "anonymous":
        sel_imgs = assigned
        assigned = row.get(None, [""])[0] if row.get(None) else ""
    return sel_imgs, assigned


def load_labels_from_active_learning(results_dir=None):
    """
    Read manually-assigned labels from the active_learning_tool CSV files.

    Returns
    -------
    image_to_label : dict[str, str]
        Mapping from image filename to its cleaned label string.
    """
    if results_dir is None:
        results_dir = ACTIVE_LEARNING_RESULTS_DIR

    # label -> set(image_name)  (same structure as app.py get_clusters)
    clusters = defaultdict(set)

    # --- single_labels.csv ---
    single_csv = os.path.join(results_dir, "single_labels.csv")
    if os.path.isfile(single_csv):
        with open(single_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img, lbl = _parse_shifted_row_single(row)
                if lbl and img:
                    clusters[_clean_label(lbl)].add(img.strip())

    # --- grid_batch_labels.csv ---
    grid_csv = os.path.join(results_dir, "grid_batch_labels.csv")
    if os.path.isfile(grid_csv):
        with open(grid_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgs_str, lbl = _parse_shifted_row_grid(row)
                if lbl and imgs_str:
                    c_label = _clean_label(lbl)
                    for img in imgs_str.split(";"):
                        if img.strip():
                            clusters[c_label].add(img.strip())

    # Flatten to image_name -> label (last-write-wins if duplicates)
    image_to_label = {}
    for label, img_set in clusters.items():
        for img in img_set:
            image_to_label[img] = label

    return image_to_label


def run_clustering(input_dir, output_dir, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading DINOv2 model (ViT-S/14)...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_paths = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))
    if not img_paths:
        print(f"No images found in '{input_dir}'. Run the cleaning first!")
        return

    valid_data = []
    features_list = []
    total = len(img_paths)
    print(f"Extracting features for {total} images...")

    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor).cpu().numpy().flatten()

        valid_data.append((filename, img))
        features_list.append(feat)
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total} images...")

    # ============================================================
    # LOAD LABELS FROM ACTIVE LEARNING TOOL
    # ============================================================
    print("\n" + "=" * 60)
    print("LOADING LABELS FROM THE ACTIVE LEARNING TOOL")
    print("=" * 60)

    image_to_label = load_labels_from_active_learning(results_dir)

    if not image_to_label:
        print("\n❌ No labels found in the results folder:")
        print(f"   {results_dir or ACTIVE_LEARNING_RESULTS_DIR}")
        print("   Make sure you have labelled some images with the web tool before running this script.")
        return

    # Build label -> numeric id mapping
    unique_labels = sorted(set(image_to_label.values()))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print(f"\nFound {len(image_to_label)} labelled images in {len(unique_labels)} categories:")
    for label in unique_labels:
        count = sum(1 for v in image_to_label.values() if v == label)
        print(f"  • {label}: {count} images")

    if len(unique_labels) < 2:
        print("\n❌ At least 2 different categories are needed to train the SVM classifier.")
        print("Nothing was saved.")
        return

    # ============================================================
    # PCA + SVM CLASSIFICATION
    # ============================================================
    feat_matrix = np.array(features_list)
    n_components = min(PCA_COMPONENTS, len(valid_data), feat_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced_feat = pca.fit_transform(feat_matrix)

    # Split data into labeled (train) and unlabeled (to predict)
    labeled_indices = []
    labels_y = []
    unlabeled_indices = []

    for i, (filename, _img) in enumerate(valid_data):
        if filename in image_to_label:
            labeled_indices.append(i)
            labels_y.append(label_to_id[image_to_label[filename]])
        else:
            unlabeled_indices.append(i)

    matched_count = len(labeled_indices)
    print(f"\nImages in the dataset matched with labels: {matched_count}/{len(valid_data)}")
    print(f"Images without a label (to be classified automatically): {len(unlabeled_indices)}")

    if matched_count == 0:
        print("\n❌ No image in the dataset matches the names of the labelled images.")
        print("Check that the images labelled in the web tool come from the same folder as the dataset.")
        return

    if len(set(labels_y)) < 2:
        print("\n❌ The matched images belong to fewer than 2 categories.")
        print("The SVM classifier needs at least 2 categories to tell apart.")
        print("Nothing was saved.")
        return

    X_train = reduced_feat[labeled_indices]
    y_train = np.array(labels_y)

    print(f"\nTraining the SVM model on {len(X_train)} labelled samples...")
    svm = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(X_train, y_train)

    # Predict ALL images (labeled + unlabeled) for consistency
    print("Automatically classifying all images...")
    predicted_labels = svm.predict(reduced_feat)

    # For manually-labeled images, use the human label (ground truth) instead of the SVM prediction
    final_labels = predicted_labels.copy()
    for idx, label_id in zip(labeled_indices, labels_y):
        final_labels[idx] = label_id

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    os.makedirs(output_dir, exist_ok=True)
    cluster_counts = {}
    for (filename, img), label_id in zip(valid_data, final_labels):
        cat_name = id_to_label[label_id]
        # Sanitize folder name (replace spaces with underscores)
        cat_folder_name = cat_name.replace(" ", "_").lower()
        cat_folder = os.path.join(output_dir, cat_folder_name)
        os.makedirs(cat_folder, exist_ok=True)
        cv2.imwrite(os.path.join(cat_folder, filename), img)
        cluster_counts[cat_name] = cluster_counts.get(cat_name, 0) + 1

    print(f"\n✅ Classification complete! Results saved in '{output_dir}'.")
    print("Final distribution:")
    for name in sorted(cluster_counts.keys()):
        print(f"  {name}: {cluster_counts[name]} images")


if __name__ == "__main__":
    clean_dir = DATA_ROOT + "/processed/train_single_bee_merged/cleaned_bees"
    output_dir = DATA_ROOT + "/processed/train_single_bee_merged/train_single_bee_clustered"
    run_clustering(clean_dir, output_dir)
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, session
import os
import csv
import random
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
app.secret_key = "active_learning_secret_key"

DATASET_FOLDER = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees" 
RESULTS_FOLDER = "results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ----------------- UTILS & DATA HANDLING ----------------- #

def get_all_images():
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    if not os.path.exists(DATASET_FOLDER):
        return []
    images = [f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(valid_extensions)]
    return sorted(images)

def log_to_csv(filename, fieldnames, row_dict):
    row_dict["user_email"] = session.get("user_email", "anonymous")
    
    filepath = os.path.join(RESULTS_FOLDER, filename)
    file_exists = os.path.isfile(filepath)
    
    actual_fieldnames = list(fieldnames)
    if "user_email" not in actual_fieldnames:
        actual_fieldnames.insert(1, "user_email")
        
    if file_exists:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
                if existing_header:
                    actual_fieldnames = existing_header
                    if "user_email" not in actual_fieldnames:
                        actual_fieldnames.append("user_email")
            except StopIteration:
                pass

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=actual_fieldnames, extrasaction='ignore')
        if not file_exists or os.path.getsize(filepath) == 0:
            writer.writeheader()
        writer.writerow(row_dict)

def clean_label(label_str):
    return " ".join(label_str.strip().split()).title()

def parse_shifted_row_single(row):
    img = str(row.get("image_name", ""))
    lbl = str(row.get("label", ""))
    if "@" in img or img == "anonymous":
        img = lbl
        lbl = str(row.get("confidence", ""))
    return img, lbl

def parse_shifted_row_grid(row):
    sel_imgs = str(row.get("selected_images", ""))
    assigned = str(row.get("assigned_label", ""))
    if "@" in sel_imgs or sel_imgs == "anonymous":
        sel_imgs = assigned
        assigned = row.get(None, [""])[0] if row.get(None) else ""
    return sel_imgs, assigned

def get_clusters():
    # builds a dictionary of clean_label -> set of image paths
    clusters = defaultdict(set)
    try:
        with open(os.path.join(RESULTS_FOLDER, "single_labels.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img, lbl = parse_shifted_row_single(row)
                if lbl and img: clusters[clean_label(lbl)].add(img.strip())
    except FileNotFoundError: pass
    
    try:
        with open(os.path.join(RESULTS_FOLDER, "grid_batch_labels.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgs_str, lbl = parse_shifted_row_grid(row)
                if lbl and imgs_str:
                    c_label = clean_label(lbl)
                    for img in imgs_str.split(";"):
                        if img.strip(): clusters[c_label].add(img.strip())
    except FileNotFoundError: pass
    
    return clusters

def get_existing_labels():
    clusters = get_clusters()
    labels = sorted(list(clusters.keys()))
    if not labels:
        labels = ["Healthy", "Varroa Mite", "Newborn", "Old / Damaged"]
    return labels

def remove_from_csv(images_to_remove, target_label):
    # removes selected outlier images from both csv files
    
    # 1. Update single_labels.csv
    single_file = os.path.join(RESULTS_FOLDER, "single_labels.csv")
    if os.path.exists(single_file):
        keep_single = []
        with open(single_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            for row in reader:
                img, lbl = parse_shifted_row_single(row)
                # skip if this is an image the user wants to remove from this target cluster
                if clean_label(lbl) == target_label and img in images_to_remove:
                    continue
                keep_single.append(row)
        with open(single_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(keep_single)

    # 2. Update grid_batch_labels.csv
    grid_file = os.path.join(RESULTS_FOLDER, "grid_batch_labels.csv")
    if os.path.exists(grid_file):
        keep_grid = []
        with open(grid_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            for row in reader:
                imgs_str, lbl = parse_shifted_row_grid(row)
                if clean_label(lbl) == target_label:
                    current_imgs = [i.strip() for i in imgs_str.split(";") if i.strip()]
                    # filter out the bad images
                    new_imgs = [i for i in current_imgs if i not in images_to_remove]
                    
                    if not new_imgs:
                        continue # drop row entirely if all images are removed
                    
                    # replace the old string safely keeping the dict keys intact
                    for k, v in row.items():
                        if v == imgs_str:
                            row[k] = ";".join(new_imgs)
                            break
                keep_grid.append(row)
        with open(grid_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(keep_grid)


# ----------------- CSS & UI TEMPLATES ----------------- #

BASE_CSS = """
<style>
    :root {
        --bg-color: #0f172a; --container-bg: #1e293b; --text-color: #f8fafc;
        --border-color: #334155; --nav-border: #334155;
        --link-color: #60a5fa; --link-hover: #93c5fd;
        --input-bg: #334155; --input-text: #f8fafc;
        --btn-blue: #3b82f6; --btn-blue-hover: #2563eb;
        --btn-green: #10b981; --btn-yellow: #f59e0b; --btn-red: #ef4444;
    }

    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; background: var(--bg-color); color: var(--text-color); }
    .container { max-width: 900px; margin: auto; background: var(--container-bg); padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
    h1, h2 { color: var(--text-color); }
    a { color: var(--link-color); text-decoration: none; font-weight: 500; }
    a:hover { text-decoration: underline; color: var(--link-hover); }
    .nav { margin-bottom: 20px; display: flex; gap: 15px; border-bottom: 1px solid var(--nav-border); padding-bottom: 12px; align-items: center; justify-content: space-between; }
    
    input[type="text"], input[type="email"], input[type="hidden"] { background: var(--input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 6px; }
    input::placeholder { color: #94a3b8; }
    
    .btn { background: var(--btn-blue); color: #fff; padding: 12px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; width: 100%; margin-top: 10px; text-decoration: none;}
    .btn:hover { background: var(--btn-blue-hover); text-decoration: none;}

    .action-group { display: flex; gap: 10px; margin-bottom: 15px; width: 100%; }
    .action-group input[type="radio"] { display: none; }
    .action-btn { flex: 1; padding: 15px 10px; text-align: center; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; color: white; opacity: 0.7; transition: 0.2s; border: 2px solid transparent; }
    .action-btn:hover { opacity: 0.9; }
    .action-group input[type="radio"]:checked + .action-btn { opacity: 1; border: 2px solid white; transform: scale(1.02); box-shadow: 0 0 10px rgba(255,255,255,0.2); }
    .btn-high { background-color: var(--btn-green); } .btn-medium { background-color: var(--btn-yellow); } .btn-low { background-color: var(--btn-red); }

    .pair-container { display: flex; gap: 20px; justify-content: center; margin: 20px 0; }
    .pair-container img { max-width: 320px; max-height: 240px; object-fit: contain; border-radius: 8px; border: 1px solid var(--border-color); }
    
    table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    th, td { border: 1px solid var(--border-color); padding: 8px 12px; text-align: left; }
    th { background: var(--input-bg); }

    .card-grid-2x6 { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 15px; }
    .grid-item-container { position: relative; cursor: pointer; display: block; }
    .grid-item-container input[type="checkbox"] { display: none; }
    .grid-item-container img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: var(--input-bg); border-radius: 8px; border: 4px solid transparent; transition: all 0.2s ease-in-out; }
    
    /* Standard checked mode (blue border) */
    .grid-item-container input[type="checkbox"]:checked + img { border: 4px solid var(--btn-blue); transform: scale(0.92); box-shadow: 0 0 15px rgba(59, 130, 246, 0.4); }
    
    /* Clean cluster mode (red border & overlay) */
    .clean-mode input[type="checkbox"]:checked + img { border: 4px solid var(--btn-red); transform: scale(0.92); box-shadow: 0 0 15px rgba(239, 68, 68, 0.6); opacity: 0.4; }
    .remove-overlay { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--btn-red); color: white; padding: 6px 10px; border-radius: 4px; font-weight: bold; opacity: 0; pointer-events: none; transition: 0.2s; z-index: 10;}
    .clean-mode input[type="checkbox"]:checked ~ .remove-overlay { opacity: 1; }
    
    .quick-label-btn { background: var(--border-color); color: var(--text-color); border: none; padding: 8px 16px; border-radius: 20px; cursor: pointer; font-size: 14px; margin: 0 8px 8px 0; transition: 0.2s; font-weight: 500; }
    .quick-label-btn:hover { background: var(--btn-blue); color: white; }

    .inspect-cluster { margin-bottom: 30px; border: 1px solid var(--border-color); padding: 15px; border-radius: 8px; background: var(--bg-color); }
    .inspect-cluster h3 { margin-top: 0; color: var(--link-hover); border-bottom: 1px solid var(--border-color); padding-bottom: 10px; display: flex; justify-content: space-between; align-items: center;}
    .inspect-images { display: flex; flex-wrap: wrap; gap: 10px; }
    .inspect-images img { width: 120px; height: 120px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border-color); }
</style>
"""


# ----------------- ROUTES ----------------- #

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        session["user_email"] = request.form.get("email").strip()
        return redirect(url_for("index"))

    user_email = session.get("user_email")

    if not user_email:
        html = f"""
        {BASE_CSS}
        <div class="container" style="max-width: 500px; text-align: center;">
            <h2>Welcome to Active Learning Hub</h2>
            <p>Please enter your email or identifier to continue. This helps track annotations per user.</p>
            <form method="POST">
                <input type="email" name="email" required placeholder="e.g. Tommaso@gmail.com" style="width: 100%; padding: 12px; margin: 15px 0; font-size: 16px;">
                <button class="btn" type="submit">Start Labeling</button>
            </form>
        </div>
        """
    else:
        html = f"""
        {BASE_CSS}
        <div class="container">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); margin-bottom: 20px; padding-bottom: 10px;">
                <h1 style="margin: 0;">Annotation Hub</h1>
                <div style="font-size: 14px; color: #94a3b8;">
                    Logged in as: <b>{user_email}</b> <a href="/logout" style="margin-left: 10px; font-size: 12px; color: var(--btn-red);">(Change)</a>
                </div>
            </div>
            
            <p>Select a survey mode or inspect current data:</p>
            <div style="display: flex; flex-direction: column; gap: 15px; margin-top: 25px;">
                <a href="/labeling" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px;">1. Single Image Labeling</a>
                <a href="/ab-test" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px;">2. A/B Cluster Similarity Survey</a>
                <a href="/grid-labeling" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px;">3. 2x6 Grid Multi-Labeling</a>
                
                <div style="border-top: 1px solid var(--border-color); margin: 5px 0;"></div>
                
                <a href="/inspect" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px; border-left: 4px solid var(--btn-yellow);">4. Inspect Clusters</a>
                <a href="/clean" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px; border-left: 4px solid var(--btn-red);">5. Clean Clusters (Remove Outliers)</a>
                <a href="/results" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px;">6. Annotation Dashboard Metrics</a>

                <div style="border-top: 1px solid var(--border-color); margin: 5px 0;"></div>

                <a href="/al-labeling" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px; border-left: 4px solid var(--btn-green);">7. Active Learning Labeling</a>
                <a href="/al-status" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px; border-left: 4px solid var(--btn-green);">8. Active Learning Status</a>
                <a href="/al-clusters" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px; border-left: 4px solid var(--btn-green);">9. Active Learning Clusters</a>
                <a href="/al-grid" style="font-size: 18px; padding: 10px; background: var(--input-bg); border-radius: 6px; border-left: 4px solid var(--btn-green);">10. Active Learning Grid (12 bees, several classes at once)</a>
            </div>
        </div>
        """
    return render_template_string(html)

@app.route("/logout")
def logout():
    session.pop("user_email", None)
    return redirect(url_for("index"))

@app.route("/get_image/<filename>")
def serve_image(filename):
    return send_from_directory(DATASET_FOLDER, filename)


@app.route("/labeling", methods=["GET", "POST"])
def single_labeling():
    if not session.get("user_email"): return redirect(url_for("index"))
    images = get_all_images()
    if not images: return f"No images found in {DATASET_FOLDER}."

    if request.method == "POST":
        image_name = request.form.get("image_name")
        label = clean_label(request.form.get("label", ""))
        confidence = request.form.get("confidence", "High")
        log_to_csv("single_labels.csv", ["timestamp", "image_name", "label", "confidence"],
                   {"timestamp": datetime.now().isoformat(), "image_name": image_name, "label": label, "confidence": confidence})
        return redirect(url_for("single_labeling"))

    target_image = random.choice(images)
    buttons_html = "".join([f'<button type="button" class="quick-label-btn" onclick="document.getElementById(\'single_label_input\').value = \'{lbl}\'">{lbl}</button>' for lbl in get_existing_labels()])

    html = f"""
    {BASE_CSS}
    <div class="container">
        <div class="nav"><a href="/">&larr; Dashboard</a><span style="font-size: 14px; color: #94a3b8;">User: {session.get('user_email')}</span></div>
        <h2>Single Sample Identification</h2>
        <div style="text-align: center; margin: 20px 0;"><img src="/get_image/{target_image}" style="max-width: 400px; max-height: 300px; border-radius: 8px;"></div>
        <form method="POST">
            <input type="hidden" name="image_name" value="{target_image}">
            <label><b>Quick Labels:</b></label><br><div style="margin: 10px 0;">{buttons_html}</div>
            <label><b>Class / Description:</b></label><br><input type="text" id="single_label_input" name="label" required placeholder="e.g. Healthy Honeybee" style="width: 100%; padding: 12px; margin: 8px 0 20px; font-size:16px;"><br>
            <label><b>Confidence:</b></label><br>
            <div class="action-group">
                <input type="radio" id="conf-high" name="confidence" value="High" required><label for="conf-high" class="action-btn btn-high">High</label>
                <input type="radio" id="conf-med" name="confidence" value="Medium"><label for="conf-med" class="action-btn btn-medium">Medium</label>
                <input type="radio" id="conf-low" name="confidence" value="Low"><label for="conf-low" class="action-btn btn-low">Low</label>
            </div>
            <button class="btn" type="submit">Submit & Next</button>
        </form>
    </div>
    """
    return render_template_string(html)


@app.route("/ab-test", methods=["GET", "POST"])
def ab_test():
    if not session.get("user_email"): return redirect(url_for("index"))
    images = get_all_images()
    if len(images) < 2: return f"Need at least 2 images in {DATASET_FOLDER}."

    if request.method == "POST":
        img_a, img_b = request.form.get("image_a"), request.form.get("image_b")
        similarity, notes = request.form.get("similarity"), request.form.get("notes", "").strip()
        log_to_csv("ab_comparisons.csv", ["timestamp", "image_a", "image_b", "similarity", "notes"],
                   {"timestamp": datetime.now().isoformat(), "image_a": img_a, "image_b": img_b, "similarity": similarity, "notes": notes})
        return redirect(url_for("ab_test"))

    img_a, img_b = random.sample(images, 2)
    html = f"""
    {BASE_CSS}
    <div class="container">
        <div class="nav"><a href="/">&larr; Dashboard</a><span style="font-size: 14px; color: #94a3b8;">User: {session.get('user_email')}</span></div>
        <h2>A/B Contrastive Similarity Survey</h2>
        <div class="pair-container">
            <div style="text-align: center;"><img src="/get_image/{img_a}"><br><small>Sample A</small></div>
            <div style="text-align: center;"><img src="/get_image/{img_b}"><br><small>Sample B</small></div>
        </div>
        <form method="POST">
            <input type="hidden" name="image_a" value="{img_a}"><input type="hidden" name="image_b" value="{img_b}">
            <label><b>Visual Similarity:</b></label><br>
            <div class="action-group">
                <input type="radio" id="sim-high" name="similarity" value="High" required><label for="sim-high" class="action-btn btn-high">High (Same cluster)</label>
                <input type="radio" id="sim-med" name="similarity" value="Medium"><label for="sim-med" class="action-btn btn-medium">Medium (Ambiguous)</label>
                <input type="radio" id="sim-low" name="similarity" value="Low"><label for="sim-low" class="action-btn btn-low">Low (Distinct)</label>
            </div>
            <label><b>Notes (optional):</b></label><br><input type="text" name="notes" placeholder="e.g. identical pose" style="width: 100%; padding: 12px; margin: 8px 0 20px; font-size:16px;"><br>
            <button class="btn" type="submit">Submit Comparison</button>
        </form>
    </div>
    """
    return render_template_string(html)


@app.route("/grid-labeling", methods=["GET", "POST"])
def grid_labeling():
    if not session.get("user_email"): return redirect(url_for("index"))
    images = get_all_images()
    if len(images) < 12: return f"Need at least 12 images in {DATASET_FOLDER} for a 2x6 grid."

    if request.method == "POST":
        selected_samples = request.form.getlist("selected_images")
        batch_label = clean_label(request.form.get("batch_label", ""))
        if selected_samples and batch_label:
            log_to_csv("grid_batch_labels.csv", ["timestamp", "selected_images", "assigned_label"],
                       {"timestamp": datetime.now().isoformat(), "selected_images": ";".join(selected_samples), "assigned_label": batch_label})
        return redirect(url_for("grid_labeling"))

    grid_images = random.sample(images, 12)
    items_html = "".join([f'<label class="grid-item-container"><input type="checkbox" name="selected_images" value="{img}"><img src="/get_image/{img}"></label>' for img in grid_images])
    buttons_html = "".join([f'<button type="button" class="quick-label-btn" onclick="document.getElementById(\'batch_label\').value = \'{lbl}\'">{lbl}</button>' for lbl in get_existing_labels()])

    html = f"""
    {BASE_CSS}
    <div class="container" style="max-width: 1100px;">
        <div class="nav"><a href="/">&larr; Dashboard</a><span style="font-size: 14px; color: #94a3b8;">User: {session.get('user_email')}</span></div>
        <h2>2x6 Grid Batch Tagging</h2>
        <p>Click directly on the images that belong together to select them.</p>
        <form method="POST">
            <div class="card-grid-2x6">{items_html}</div><br>
            <label><b>Quick Labels:</b></label><br><div style="margin: 10px 0;">{buttons_html}</div>
            <label><b>Batch Label for Selected Samples:</b></label><br><input type="text" id="batch_label" name="batch_label" required style="width: 100%; padding: 12px; margin: 8px 0 20px; font-size:16px;"><br>
            <button class="btn" type="submit">Submit Batch</button>
        </form>
    </div>
    """
    return render_template_string(html)


@app.route("/inspect")
def inspect():
    if not session.get("user_email"): return redirect(url_for("index"))
    clusters = get_clusters()
    
    clusters_html = ""
    if not clusters:
        clusters_html = "<p>No labeled data found yet.</p>"
    else:
        for label, img_set in sorted(clusters.items()):
            example_imgs = list(img_set)[:10]
            imgs_html = "".join([f'<img src="/get_image/{img}" title="{img}">' for img in example_imgs])
            clusters_html += f"""
            <div class="inspect-cluster">
                <h3><span>Label: {label} <span style="font-size: 14px; color: #94a3b8; font-weight: normal;">({len(img_set)} total images)</span></span>
                    <a href="/clean?label={label}" style="font-size: 14px; background: var(--btn-red); color: white; padding: 4px 8px; border-radius: 4px;">Clean Cluster</a>
                </h3>
                <div class="inspect-images">{imgs_html}</div>
            </div>
            """

    html = f"""
    {BASE_CSS}
    <div class="container" style="max-width: 1100px;">
        <div class="nav"><a href="/">&larr; Dashboard</a><span style="font-size: 14px; color: #94a3b8;">User: {session.get('user_email')}</span></div>
        <h2>Cluster Inspector</h2>
        <p>Overview of unique labels assigned so far.</p>
        <div style="margin-top: 25px;">{clusters_html}</div>
    </div>
    """
    return render_template_string(html)


@app.route("/clean", methods=["GET", "POST"])
def clean():
    if not session.get("user_email"): return redirect(url_for("index"))
    
    if request.method == "POST":
        target_label = request.form.get("target_label")
        images_to_remove = request.form.getlist("selected_images")
        if target_label and images_to_remove:
            remove_from_csv(images_to_remove, target_label)
        return redirect(url_for("clean", label=target_label))
        
    target_label = request.args.get("label")
    clusters = get_clusters()
    
    # 1. If no label selected, show choice screen
    if not target_label or target_label not in clusters:
        buttons_html = "".join([f'<a href="/clean?label={lbl}" class="btn" style="display:inline-block; margin:5px 5px 10px 0; width:auto; font-weight:normal;">{lbl} <span style="color:#cbd5e1; font-size:14px;">({len(clusters[lbl])})</span></a>' for lbl in sorted(clusters.keys())])
        if not buttons_html: buttons_html = "<p>No data to clean yet.</p>"
        html = f"""
        {BASE_CSS}
        <div class="container">
            <div class="nav"><a href="/">&larr; Dashboard</a><span style="font-size: 14px; color: #94a3b8;">User: {session.get('user_email')}</span></div>
            <h2>Data Cleaning</h2>
            <p>Select a cluster to review. You will be shown random samples to filter out misclassifications.</p>
            <div style="margin-top: 20px;">{buttons_html}</div>
        </div>
        """
        return render_template_string(html)
        
    # 2. Label is selected, show 2x6 removal grid
    all_imgs = list(clusters[target_label])
    if not all_imgs:
        return redirect(url_for("clean"))
        
    sample_size = min(12, len(all_imgs))
    grid_images = random.sample(all_imgs, sample_size)
    
    items_html = ""
    for img in grid_images:
        items_html += f"""
        <label class="grid-item-container clean-mode">
            <input type="checkbox" name="selected_images" value="{img}">
            <img src="/get_image/{img}">
            <div class="remove-overlay">REMOVE</div>
        </label>
        """
        
    html = f"""
    {BASE_CSS}
    <div class="container" style="max-width: 1100px;">
        <div class="nav"><a href="/">&larr; Dashboard</a><span style="font-size: 14px; color: #94a3b8;">User: {session.get('user_email')}</span></div>
        <h2>Cleaning Cluster: <span style="color:var(--btn-red);">{target_label}</span></h2>
        <p>Click on the images that <b>DO NOT</b> belong to '{target_label}'. They will be permanently removed from this label in the CSV files.</p>
        <form method="POST">
            <input type="hidden" name="target_label" value="{target_label}">
            <div class="card-grid-2x6">
                {items_html}
            </div>
            <br>
            <button class="btn" style="background:var(--btn-red);" type="submit">Remove Selected Outliers</button>
            <a href="/clean?label={target_label}" class="btn" style="background:var(--input-bg); color:white; display:block; text-align:center; margin-top:10px;">Skip / Show Different Images</a>
        </form>
    </div>
    """
    return render_template_string(html)


@app.route("/results")
def results():
    if not session.get("user_email"): return redirect(url_for("index"))
    def count_rows(filename):
        path = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(path): return 0
        with open(path, "r", encoding="utf-8") as f: return max(0, sum(1 for _ in f) - 1)

    html = f"""
    {BASE_CSS}
    <div class="container">
        <div class="nav"><a href="/">&larr; Dashboard</a><span style="font-size: 14px; color: #94a3b8;">User: {session.get('user_email')}</span></div>
        <h2>Annotation Metrics</h2>
        <table>
            <tr><th>Task</th><th>CSV Target</th><th>Total Completed Records</th></tr>
            <tr><td>Single Labeling</td><td>single_labels.csv</td><td>{count_rows("single_labels.csv")}</td></tr>
            <tr><td>A/B Similarity Testing</td><td>ab_comparisons.csv</td><td>{count_rows("ab_comparisons.csv")}</td></tr>
            <tr><td>2x6 Batch Selection</td><td>grid_batch_labels.csv</td><td>{count_rows("grid_batch_labels.csv")}</td></tr>
        </table>
    </div>
    """
    return render_template_string(html)


# ----------------- ACTIVE LEARNING (bee_al) ----------------- #
# Adds /al-labeling, /al-clusters, /al-status, /al-refresh, /al-retrain and /al-image. The routes above are untouched
# and keep working even if bee_al or its dependencies are missing.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from bee_al.hub import create_blueprint
    app.register_blueprint(create_blueprint(BASE_CSS))
except ImportError as e:
    print(f"Active learning pages disabled: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=False)
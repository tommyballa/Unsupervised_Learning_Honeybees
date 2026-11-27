"""Exports the clusters of an already frozen snapshot using hardlinks instead of copies.

On the NTFS (fuse) disk shutil.copy2 runs at ~2 bees/s: 174k bees would take ~20 hours.
Uses the same code as the hub (ClusterExporter._run); config_id stays that of the snapshot.
    venv/bin/python logs/export_fase1_hardlink.py <snapshot_name>
"""
import json
import sys
import threading
import time
from types import SimpleNamespace

from bee_al.config import load_config
from bee_al.export import ClusterExporter
from bee_al.snapshot import snapshot_path

name = sys.argv[1]
cfg = load_config("config/al_config.yaml")
config_id, config_dict = cfg.config_id, cfg.to_dict()
snapshot_dir = snapshot_path(cfg, name)
assert json.load(open(snapshot_dir / "meta.json"))["config_id"] == config_id, "config changed after the snapshot"
assert not (cfg.clusters_dir / name).exists(), "export already present"

cfg.export.mode = "hardlink"
exporter = ClusterExporter(SimpleNamespace(cfg=cfg, config_path=None))
exporter.progress = {"name": name, "copied": 0, "total": 0, "started": time.time()}
annotator = json.load(open(snapshot_dir / "meta.json"))["annotator"]
t = threading.Thread(target=exporter._run, args=(name, annotator, snapshot_dir))
t.start()
while t.is_alive():
    t.join(60)
    p = exporter.progress
    print(time.strftime("%H:%M:%S"), f"{p['copied']} / {p['total']}", flush=True)
if exporter.state == "failed":
    sys.exit(f"Export failed: {exporter.error}")

# The export's config.yaml and config_id stay those of the phase (mode: copy); meta.mode says hardlink
out = cfg.clusters_dir / name
meta = json.load(open(out / "meta.json"))
meta.update(config_id=config_id, config_file="al_config.yaml")
json.dump(meta, open(out / "meta.json", "w"), indent=2, ensure_ascii=False)
import yaml
yaml.safe_dump(config_dict, open(out / "config.yaml", "w"), sort_keys=False, allow_unicode=True)
print("Done:", out, meta["clusters"], "missing files:", meta["missing_files"])

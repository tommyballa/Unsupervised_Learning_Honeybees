"""Esporta i cluster di uno snapshot già congelato con hardlink invece che copie.

Sul disco NTFS (fuse) shutil.copy2 va a ~2 api/s: 174k api richiederebbero ~20 ore.
Usa lo stesso codice dell'hub (ClusterExporter._run); config_id resta quello dello snapshot.
    venv/bin/python logs/export_fase1_hardlink.py <nome_snapshot>
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
assert json.load(open(snapshot_dir / "meta.json"))["config_id"] == config_id, "config cambiata dopo lo snapshot"
assert not (cfg.clusters_dir / name).exists(), "export già presente"

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
    sys.exit(f"Export fallito: {exporter.error}")

# Il config.yaml e il config_id dell'export restano quelli della fase (mode: copy); meta.mode dice hardlink
out = cfg.clusters_dir / name
meta = json.load(open(out / "meta.json"))
meta.update(config_id=config_id, config_file="al_config.yaml")
json.dump(meta, open(out / "meta.json", "w"), indent=2, ensure_ascii=False)
import yaml
yaml.safe_dump(config_dict, open(out / "config.yaml", "w"), sort_keys=False, allow_unicode=True)
print("Fatto:", out, meta["clusters"], "file mancanti:", meta["missing_files"])

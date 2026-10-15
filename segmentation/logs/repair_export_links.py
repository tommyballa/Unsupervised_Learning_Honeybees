"""Ricrea gli hardlink mancanti di un export di cluster e ricontrolla che ci siano tutti.
    venv/bin/python logs/repair_export_links.py <cartella export>
"""
import csv
import os
import sys
import time

out = sys.argv[1]
rows = [r for r in csv.DictReader(open(os.path.join(out, "clusters.csv"), encoding="utf-8")) if r["file"]]
present = {d: set(os.listdir(os.path.join(out, d))) for d in {r["file"].split("/")[0] for r in rows}}
missing = [r for r in rows if r["file"].split("/")[1] not in present[r["file"].split("/")[0]]]
print(time.strftime("%H:%M:%S"), f"mancanti {len(missing)} su {len(rows)}", flush=True)
for i, r in enumerate(missing):
    os.link(r["original_path"], os.path.join(out, r["file"]))
    if i % 5000 == 0:
        print(time.strftime("%H:%M:%S"), i, flush=True)
present = {d: set(os.listdir(os.path.join(out, d))) for d in present}
still = [r for r in rows if r["file"].split("/")[1] not in present[r["file"].split("/")[0]]]
bad = [r for r in rows if not os.path.isfile(os.path.join(out, r["file"]))]
print(time.strftime("%H:%M:%S"), f"dopo la riparazione: mancanti in listdir {len(still)}, non apribili {len(bad)}")

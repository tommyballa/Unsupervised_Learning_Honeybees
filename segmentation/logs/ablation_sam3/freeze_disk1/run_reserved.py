"""Esegue uno script Python dopo aver prenotato quasi tutta la memoria libera della GPU, e la tiene fino alla fine.

Serve perche' il server e' condiviso: Ultralytics restituisce la cache alla GPU a ogni epoca (_clear_memory),
gli altri processi se la prendono e al picco successivo il nostro training va in out of memory. Cosi' invece
la memoria resta nostra: se qualcuno prova a crescere sulla stessa GPU e' lui a non trovarla.

La GPU va scelta con CUDA_VISIBLE_DEVICES (la coda lo fa), quindi dentro lo script e' sempre il device 0.

Uso: CUDA_VISIBLE_DEVICES=<gpu> python run_reserved.py <script.py> [argomenti...]
     python run_reserved.py -c "<codice>" [argomenti...]
Variabili: RESERVE_GB (default: tutta la memoria libera meno MARGIN_GB, al massimo MAX_GB)
"""
import gc
import os
import runpy
import sys
import time

# con expandable_segments la memoria liberata torna alla GPU: qui deve restare nella cache di PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
import torch

MARGIN_GB = float(os.environ.get("MARGIN_GB", 1.0))   # lasciati liberi per il contesto CUDA e il resto
MAX_GB = float(os.environ.get("MAX_GB", 44.0))         # picco misurato 35-41 GB: sopra non serve
SMALL_MB = 512                                         # cache dei tensori piccoli (< 1 MB), che usano un'altra riserva

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "scegliere la GPU con CUDA_VISIBLE_DEVICES"
torch.cuda.set_device(0)
free, total = torch.cuda.mem_get_info(0)
gb = float(os.environ.get("RESERVE_GB", 0)) or min(free / 2**30 - MARGIN_GB - SMALL_MB / 1024, MAX_GB)
# un unico blocco grande: PyTorch lo tiene in cache e lo divide per le allocazioni del training
big = torch.empty(int(gb * 2**30), dtype=torch.uint8, device="cuda")
small = [torch.empty(512 * 1024, dtype=torch.uint8, device="cuda") for _ in range(SMALL_MB * 2)]
del big, small
print(f"{time.strftime('%F %T')} run_reserved: GPU {os.environ['CUDA_VISIBLE_DEVICES']}, "
      f"prenotati {torch.cuda.memory_reserved(0) / 2**30:.1f} GB (liberi prima {free / 2**30:.1f} di "
      f"{total / 2**30:.1f} GB)", flush=True)

# la cache non va mai restituita alla GPU
torch.cuda.empty_cache = lambda: None
from ultralytics.engine.trainer import BaseTrainer
BaseTrainer._clear_memory = lambda self, threshold=None: gc.collect()

if sys.argv[1] == "-c":
    code = sys.argv[2]
    sys.argv = ["-c"] + sys.argv[3:]
    exec(compile(code, "<run_reserved -c>", "exec"), {"__name__": "__main__"})
else:
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")

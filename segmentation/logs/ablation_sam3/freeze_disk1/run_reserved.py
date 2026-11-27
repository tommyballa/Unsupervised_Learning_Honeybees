"""Runs a Python script after reserving almost all the free GPU memory, and holds it until the end.

Needed because the server is shared: Ultralytics returns the cache to the GPU at every epoch (_clear_memory),
other processes grab it and at the next peak our training runs out of memory. This way the memory
stays ours: if someone tries to grow on the same GPU, they are the one who does not find it.

The GPU must be chosen with CUDA_VISIBLE_DEVICES (the queue does this), so inside the script it is always device 0.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python run_reserved.py <script.py> [arguments...]
       python run_reserved.py -c "<code>" [arguments...]
Variables: RESERVE_GB (default: all the free memory minus MARGIN_GB, at most MAX_GB)
"""
import gc
import os
import runpy
import sys
import time

# with expandable_segments freed memory goes back to the GPU: here it must stay in the PyTorch cache
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
import torch

MARGIN_GB = float(os.environ.get("MARGIN_GB", 1.0))   # left free for the CUDA context and everything else
MAX_GB = float(os.environ.get("MAX_GB", 44.0))         # measured peak 35-41 GB: more is not needed
SMALL_MB = 512                                         # cache for small tensors (< 1 MB), which use a separate pool

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "choose the GPU with CUDA_VISIBLE_DEVICES"
torch.cuda.set_device(0)
free, total = torch.cuda.mem_get_info(0)
gb = float(os.environ.get("RESERVE_GB", 0)) or min(free / 2**30 - MARGIN_GB - SMALL_MB / 1024, MAX_GB)
# a single large block: PyTorch keeps it in cache and splits it for the training allocations
big = torch.empty(int(gb * 2**30), dtype=torch.uint8, device="cuda")
small = [torch.empty(512 * 1024, dtype=torch.uint8, device="cuda") for _ in range(SMALL_MB * 2)]
del big, small
print(f"{time.strftime('%F %T')} run_reserved: GPU {os.environ['CUDA_VISIBLE_DEVICES']}, "
      f"reserved {torch.cuda.memory_reserved(0) / 2**30:.1f} GB (free before: {free / 2**30:.1f} of "
      f"{total / 2**30:.1f} GB)", flush=True)

# the cache must never be returned to the GPU
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

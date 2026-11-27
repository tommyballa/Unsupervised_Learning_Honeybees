"""
Inference cost of a single-bee model (backbone ablation, see ablation.py).

Loads the best checkpoint saved by training (same path as plot_heatmaps.py) and
measures, on the same computation as run_inference (model in fp32 + map_postprocess
+ map resized back to 512x512) with real validation-set images:

  - model parameters and memory (weights + buffers + tensors such as memory bank
    and covariance), in fp32;
  - GPU latency at batch 1 (median over --iters after warm-up) and throughput at
    batch --batch;
  - peak GPU memory at batch 1;
  - GFLOP per image (torch.utils.flop_counter: convolutions and matmuls,
    independent of GPU load; torch.cdist, which the counter does not see,
    is counted separately as a matmul, 2·n·m·d: it is PatchCore's kNN search;
    blur and elementwise operations are not counted);
  - with --cpu, CPU latency at batch 1 (--threads threads).

The ABL_* variables select the configuration as for training, e.g.:

    ABL_BACKBONE=resnet18 CUDA_VISIBLE_DEVICES=1 python experiments/single_bee/measure_efficiency.py cfa

The result goes to logs/ablation/efficiency/<model>_<tag|default>.json. For
comparable timings the GPU must be free.
"""

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "single_bee"))

import ablation  # noqa: E402
from plot_heatmaps import load_best  # noqa: E402


def collect_tensors(obj, seen=None, depth=0):
    """Tensors reachable from obj (modules, attributes, dicts, lists), without duplicates."""
    seen = {} if seen is None else seen
    if depth > 6 or id(obj) in seen:
        return seen
    if torch.is_tensor(obj):
        seen[id(obj)] = obj
        return seen
    if isinstance(obj, torch.nn.Module):
        seen[id(obj)] = None
        for t in list(obj.parameters()) + list(obj.buffers()):
            seen.setdefault(id(t), t)
        items = vars(obj).items()
    elif isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif (hasattr(obj, "__dict__") and not isinstance(obj, (type, types.FunctionType, types.MethodType, types.ModuleType))
          and type(obj).__module__.split(".")[0] not in ("builtins", "torch", "numpy")):
        seen[id(obj)] = None
        items = vars(obj).items()
    else:
        return seen
    for _, value in items:
        collect_tensors(value, seen, depth + 1)
    return seen


def model_size(model):
    tensors = [t for t in collect_tensors(model).values() if t is not None and t.is_floating_point()]
    numel = sum(t.numel() for t in tensors)
    return numel, numel * 4 / 2**20


def infer(mod, model, images):
    maps, scores = model(images)
    maps = maps.float().reshape(images.shape[0], 1, *maps.shape[-2:])
    postprocess = getattr(mod, "map_postprocess", None)
    if postprocess is not None:
        maps = postprocess(maps)
    if maps.shape[-2:] != tuple(mod.GT_SIZE):
        maps = F.interpolate(maps, size=tuple(mod.GT_SIZE), mode="bilinear", align_corners=False)
    return maps, scores


@torch.no_grad()
def time_it(mod, model, images, iters, warmup, device):
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        infer(mod, model, images)
    sync()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        infer(mod, model, images)
        sync()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def load_images(mod, n):
    from moviad.datasets.varroa.varroa_dataset import VarroaDataset
    from moviad.utilities.configurations import Split

    dataset = VarroaDataset(mod.dataset_args(mod.VAL_DIR), "varroa", Split.TEST)
    images = next(iter(DataLoader(dataset, batch_size=n, shuffle=False, num_workers=4)))[0]
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--cpu", action="store_true", help="also measure CPU latency")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-iters", type=int, default=10)
    args = parser.parse_args()

    import importlib
    mod = importlib.import_module(f"main_{args.model}")
    mod.set_seed(mod.SEED)
    # PatchCore backbones already compared outside the ablation (PATCHCORE_BACKBONE)
    tag = ablation.tag() or os.environ.get("PATCHCORE_BACKBONE") or "default"
    images = load_images(mod, args.batch)
    result = {"model": args.model, "tag": tag,
              "backbone": getattr(mod, "BACKBONE", None) or getattr(mod, "ENCODER_NAME", None),
              "img_size": list(mod.IMG_SIZE)}

    device = torch.device("cuda:0")
    model, _ = load_best(mod, device)
    model.eval()
    result["params_M"], result["size_MB"] = model_size(model)
    result["params_M"] /= 1e6
    cdist_flops = [0]
    cdist = torch.cdist

    def counted_cdist(x1, x2, *a, **k):
        cdist_flops[0] += 2 * x1.shape[:-1].numel() * x2.shape[-2] * x1.shape[-1]
        return cdist(x1, x2, *a, **k)

    torch.cdist = counted_cdist
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as flops:
            infer(mod, model, images[:1].to(device))
    finally:
        torch.cdist = cdist
    result["gflops"] = (flops.get_total_flops() + cdist_flops[0]) / 1e9
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    result["gpu_latency_ms"] = 1e3 * time_it(mod, model, images[:1].to(device), args.iters, args.warmup, device)
    result["gpu_peak_MB_bs1"] = (torch.cuda.max_memory_allocated() - base) / 2**20
    result["gpu_model_MB"] = base / 2**20
    batch_time = time_it(mod, model, images.to(device), max(5, args.iters // 10), 3, device)
    result[f"gpu_throughput_bs{args.batch}"] = args.batch / batch_time
    result["gpu"] = torch.cuda.get_device_name(0)
    del model
    torch.cuda.empty_cache()

    if args.cpu:
        torch.set_num_threads(args.threads)
        cpu = torch.device("cpu")
        try:
            model, _ = load_best(mod, cpu)
            model.eval()
            # no bf16 autocast on CPU (e.g. FastFlow backbone): much slower than fp32
            for m in model.modules() if isinstance(model, torch.nn.Module) else []:
                if getattr(m, "backbone_amp", False):
                    m.backbone_amp = False
            result["cpu_latency_ms"] = 1e3 * time_it(mod, model, images[:1], args.cpu_iters, 2, cpu)
            result["cpu_threads"] = args.threads
        except Exception as e:      # some models hard-code the cuda device internally
            result["cpu_error"] = f"{type(e).__name__}: {e}"

    out_dir = mod.REPO_ROOT / "logs" / "ablation" / "efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.model}_{tag}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Extract, once, the embeddings of all bees with a frozen extractor.

Usage:
    python scripts/extract_embeddings.py                      # defaults from config/al_config.yaml
    python scripts/extract_embeddings.py --model dinov2_vitb14 --out data/embeddings_vitb14

It can be interrupted and restarted: it resumes from the missing images.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.config import load_config  # noqa: E402
from bee_al.embeddings import (  # noqa: E402
    build_torch_encoder,
    extract_embeddings,
    list_dataset,
    load_backbone,
    read_manifest,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="configuration file (default config/al_config.yaml)")
    ap.add_argument("--model", default=None, help="overrides extractor.model")
    ap.add_argument("--out", default=None, help="overrides paths.embeddings_dir")
    ap.add_argument("--device", default=None, help="overrides extractor.device")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (quick test)")
    ap.add_argument("--weights", default=None, help="weights of a re-train (e.g. models/v001/backbone.pt) instead of the pre-trained ones")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ex = cfg.extractor
    model_name = args.model or ex.model
    out_dir = cfg.resolve(args.out) if args.out else cfg.embeddings_dir

    sources = cfg.dataset_sources
    image_ids, images = list_dataset(sources)
    for prefix, folder in sources:
        n = sum(i.startswith(prefix) for i in image_ids) if prefix else len(image_ids)
        print(f"{n:6d} images in {folder}" + ("" if folder.is_dir() else "  (folder not found)"))
    if args.limit:
        image_ids, images = image_ids[:args.limit], images[:args.limit]
    if not images:
        sys.exit("No images found")
    print(f"Model {model_name}, output {out_dir}")

    backbone = load_backbone(model_name)
    if args.weights:
        import torch

        backbone.load_state_dict(torch.load(cfg.resolve(args.weights), map_location="cpu"))
    encoder = build_torch_encoder(
        backbone,
        image_size=ex.image_size,
        batch_size=ex.batch_size,
        num_workers=ex.num_workers,
        device=args.device or ex.device,
    )
    t0 = time.time()
    added = extract_embeddings(
        images,
        out_dir,
        encoder,
        meta={"model": model_name, "image_size": ex.image_size, **({"weights": args.weights} if args.weights else {})},
        shard_size=ex.shard_size,
        log=print,
        image_ids=image_ids,
    )
    manifest = read_manifest(out_dir)
    total = sum(s["count"] for s in manifest["shards"])
    print(f"Added {added} embeddings in {time.time() - t0:.1f}s. Total {total}, dim {manifest['dim']}.")
    if manifest["failed"]:
        print(f"Unreadable images skipped: {len(manifest['failed'])}")


if __name__ == "__main__":
    main()

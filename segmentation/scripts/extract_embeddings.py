"""Estrae una volta gli embedding di tutte le api con un extractor congelato.

Uso:
    python scripts/extract_embeddings.py                      # default da config/al_config.yaml
    python scripts/extract_embeddings.py --model dinov2_vitb14 --out data/embeddings_vitb14

Si può interrompere e rilanciare: riprende dalle immagini mancanti.
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
    list_images,
    load_backbone,
    read_manifest,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="file di configurazione (default config/al_config.yaml)")
    ap.add_argument("--model", default=None, help="sovrascrive extractor.model")
    ap.add_argument("--out", default=None, help="sovrascrive paths.embeddings_dir")
    ap.add_argument("--device", default=None, help="sovrascrive extractor.device")
    ap.add_argument("--limit", type=int, default=None, help="solo le prime N immagini (prova veloce)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ex = cfg.extractor
    model_name = args.model or ex.model
    out_dir = cfg.resolve(args.out) if args.out else cfg.embeddings_dir

    images = list_images(cfg.dataset_dir)
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"Nessuna immagine trovata in {cfg.dataset_dir}")
    print(f"{len(images)} immagini in {cfg.dataset_dir}")
    print(f"Modello {model_name}, output {out_dir}")

    encoder = build_torch_encoder(
        load_backbone(model_name),
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
        meta={"model": model_name, "image_size": ex.image_size},
        shard_size=ex.shard_size,
        log=print,
    )
    manifest = read_manifest(out_dir)
    total = sum(s["count"] for s in manifest["shards"])
    print(f"Aggiunti {added} embedding in {time.time() - t0:.1f}s. Totale {total}, dim {manifest['dim']}.")
    if manifest["failed"]:
        print(f"Immagini illeggibili saltate: {len(manifest['failed'])}")


if __name__ == "__main__":
    main()

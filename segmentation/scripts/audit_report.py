"""Stime corrette e indicatori di colonia per uno snapshot con audit.

Uso:
    python scripts/audit_report.py --snapshot 2026-10-05_round42
    python scripts/audit_report.py --snapshot NAME --anomaly-scores scores.csv   # CSV image_id,score

Scrive in ``<results_dir>/snapshots/<snapshot>/report/``: strata.csv, prevalence.csv, groups.csv,
colony_indicators.json/.csv, agreement.json e report.md. Non serve che l'hub sia acceso.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.config import load_config  # noqa: E402
from bee_al.report import REPORT_DIR, build_report  # noqa: E402
from bee_al.snapshot import snapshot_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="nome dello snapshot")
    ap.add_argument("--anomaly-scores", default=None, help="CSV image_id,score per l'indicatore di anomaly detection")
    ap.add_argument("--config", default=None, help="file di configurazione (default config/al_config.yaml)")
    ap.add_argument("--n-boot", type=int, default=None, help="repliche bootstrap (default colony.n_boot)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        folder = snapshot_path(cfg, args.snapshot)
        report = build_report(folder, cfg, anomaly_scores_path=args.anomaly_scores, n_boot=args.n_boot, force=True)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Report non creato: {e}")
    est = report["estimates"]
    print(f"Audit: {est['audit']['answered']} risposte su {est['audit']['sampled']} api campionate"
          + ("" if est["complete"] else " (audit non completo)"))
    for p in est["prevalence"]:
        if not p["reject"]:
            print(f"  {p['class']:<28} classify and count {100 * (p['cc_share'] or 0):5.1f}%   stima {100 * p['share']:5.1f}% "
                  f"({100 * p['ci_boot_low']:.1f} – {100 * p['ci_boot_high']:.1f})")
    print(f"Report scritto in {folder / REPORT_DIR}")


if __name__ == "__main__":
    main()

"""Location of the bee dataset on disk.

Every script resolves its default input/output folders under ``DATA_ROOT``.
Set the ``BEES_DATA_ROOT`` environment variable to point it at your copy of
the dataset; by default it is ``<repository root>/data/DatasetApi_Ceschi``.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = os.environ.get("BEES_DATA_ROOT", str(_REPO_ROOT / "data" / "DatasetApi_Ceschi"))

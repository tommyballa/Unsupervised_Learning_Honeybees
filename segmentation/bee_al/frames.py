"""Link from each bee to its source frame, from the extraction and cleaning manifests.

``extract_anomaly_dataset.py`` numbers the crops with a counter that increases by 1 for each bee,
frame after frame, and records in ``raw_bees/_extraction_manifest.json`` how many bees it saved
for each frame (``processed_sources``, in processing order) and the next free id
(``next_bee_id``). Summing the counts in file order gives, for each frame, the
``[start, end)`` range of its bees' ids. ``dataset_cleaning.py`` discards some crops
but does not renumber them, so the cleaned file name is enough to trace the frame.
"""
from __future__ import annotations

import bisect
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .config import ID_SEPARATOR, ALConfig

BEE_ID_RE = re.compile(r"^bee_(\d+)_")
EXTRACTION_MANIFEST = "_extraction_manifest.json"
CLEANING_MANIFEST = "_cleaning_manifest.json"


@dataclass
class FrameMap:
    """Bee id ranges for each frame of a dataset folder."""

    frames: list[str] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    reliable: bool = True
    path: str = ""

    def frame_of(self, filename: str) -> Optional[str]:
        """Frame the crop ``bee_<id>_...`` comes from.

        ``None`` if the name does not have that form, if the id is outside every range
        or if the map is unreliable: in that case there is no guessing.
        """
        m = BEE_ID_RE.match(Path(filename).name)
        if m is None or not self.reliable:
            return None
        bee_id = int(m.group(1))
        i = bisect.bisect_right(self.starts, bee_id) - 1
        if i < 0 or bee_id >= self.ends[i]:
            return None
        return self.frames[i]

    def raw_bees_per_frame(self) -> dict[str, int]:
        """Raw bees (before cleaning) extracted from each frame."""
        return {f: e - s for f, s, e in zip(self.frames, self.starts, self.ends)}


def load_frame_map(manifest_path: Union[str, Path]) -> FrameMap:
    """Build the id ranges by walking ``processed_sources`` in file order.

    If the sum of the counts does not match ``next_bee_id`` the numbering is not the expected
    one (manifest edited by hand, mixed extractions, ...): the map is marked
    ``reliable = False`` with a warning and ``frame_of`` no longer returns anything.
    """
    path = Path(manifest_path)
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    fmap = FrameMap(path=str(path))
    cursor = 0
    for frame, info in manifest.get("processed_sources", {}).items():
        n = int(info["bees"] if isinstance(info, dict) else info)
        fmap.frames.append(frame)
        fmap.starts.append(cursor)
        cursor += n
        fmap.ends.append(cursor)
    expected = manifest.get("next_bee_id")
    if expected is None or int(expected) != cursor:
        fmap.reliable = False
        warnings.warn(
            f"{path}: the sum of bees per frame ({cursor}) does not match next_bee_id ({expected}); "
            "the bee -> frame link for this folder is not used.",
            stacklevel=2,
        )
    return fmap


def discards_per_frame(cleaning_manifest_path: Union[str, Path], frame_map: FrameMap) -> dict[str, dict[str, Any]]:
    """For each frame: crops accepted and discarded by the cleaning step, and the discard reasons.

    Returns ``{frame: {"accepted": n, "discarded": n, "reasons": {reason: n}}}``;
    an empty dict if the cleaning manifest does not exist.
    """
    path = Path(cleaning_manifest_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    reasons = manifest.get("discard_reasons", {})
    out: dict[str, dict[str, Any]] = {}
    for filename, status in manifest.get("processed_files", {}).items():
        frame = frame_map.frame_of(filename)
        if frame is None:
            continue
        row = out.setdefault(frame, {"accepted": 0, "discarded": 0, "reasons": {}})
        if status == "accepted":
            row["accepted"] += 1
        else:
            row["discarded"] += 1
            reason = reasons.get(filename, "unknown")
            row["reasons"][reason] = row["reasons"].get(reason, 0) + 1
    return out


def dataset_folders(cfg: ALConfig) -> dict[str, Path]:
    """Dataset folders by name: the one in ``paths.dataset_dirs``, or the folder name if there is only one."""
    if cfg.paths.dataset_dirs:
        return {name: cfg.resolve(d) for name, d in cfg.paths.dataset_dirs.items()}
    folder = cfg.dataset_dir
    return {folder.parent.name if folder.name == "cleaned_bees" else folder.name: folder}


def dataset_of(image_id: str, cfg: ALConfig) -> str:
    """Name of the dataset folder an image_id belongs to."""
    if cfg.paths.dataset_dirs:
        name, sep, _ = image_id.partition(ID_SEPARATOR)
        if sep and name in cfg.paths.dataset_dirs:
            return name
        return ""
    return next(iter(dataset_folders(cfg)))


def extraction_manifest_path(cfg: ALConfig, dataset: str, folder: Path) -> Path:
    """Extraction manifest: the one in ``frames.manifests``, otherwise ``<cleaned folder>/../raw_bees/``."""
    if dataset in cfg.frames.manifests:
        return cfg.resolve(cfg.frames.manifests[dataset])
    return folder.parent / "raw_bees" / EXTRACTION_MANIFEST


def load_frame_maps(cfg: ALConfig) -> dict[str, Optional[FrameMap]]:
    """One ``FrameMap`` per dataset folder; ``None`` where the manifest cannot be found."""
    maps: dict[str, Optional[FrameMap]] = {}
    for dataset, folder in dataset_folders(cfg).items():
        path = extraction_manifest_path(cfg, dataset, folder)
        maps[dataset] = load_frame_map(path) if path.exists() else None
    return maps


class FrameGrouper:
    """Session and hive of every frame, according to the ``frames`` config.

    Precedence for the session: the regex ``session`` group, then the ``session_dirs``
    folder that contains the frame, then the dataset folder name. For
    the hive: the regex ``colony`` group, then ``frames.colony``. A file name present
    in several session folders is ambiguous and is assigned to none of them.
    """

    def __init__(self, cfg: ALConfig) -> None:
        self.pattern = cfg.frames.frame_pattern
        self.colony = cfg.frames.colony
        self.session_of: dict[str, str] = {}
        self.ambiguous: set[str] = set()
        self.missing_dirs: list[str] = []
        for session, folder in cfg.frames.session_dirs.items():
            root = cfg.resolve(folder)
            if not root.is_dir():
                self.missing_dirs.append(str(root))
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if self.session_of.get(path.name, session) != session:
                    self.ambiguous.add(path.name)
                self.session_of[path.name] = session
        for name in self.ambiguous:
            del self.session_of[name]
        if self.ambiguous:
            warnings.warn(f"{len(self.ambiguous)} frames appear in several session folders (e.g. {sorted(self.ambiguous)[0]}): "
                          "they are left without a session.", stacklevel=2)

    def groups(self, frame: str, dataset: str) -> tuple[str, str]:
        """``(session, colony)`` of a frame."""
        session, colony = frame_groups(frame, dataset, self.pattern)
        if session == dataset and frame in self.session_of:
            session = self.session_of[frame]
        return session, colony or self.colony


def frame_groups(frame: str, dataset: str, pattern: str = "") -> tuple[str, str]:
    """Session and hive of a frame: ``(session, colony)``.

    ``pattern`` is a regex with the named groups ``session`` and/or ``colony``, searched in the
    frame name. Without a regex, or if it does not match, the session is the dataset folder
    name and the hive is empty: nothing is invented from file names alone.
    """
    session, colony = dataset, ""
    if pattern and frame:
        m = re.search(pattern, frame)
        if m:
            groups = m.groupdict()
            session = groups.get("session") or session
            colony = groups.get("colony") or ""
    return session, colony

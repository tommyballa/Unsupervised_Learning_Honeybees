"""Typed configuration, read from ``config/al_config.yaml``."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "al_config.yaml"

UNCERTAINTIES = ("margin", "entropy", "least_confidence")
ID_SEPARATOR = "__"  # between folder name and file name in image_ids


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Invalid config: {msg}")


@dataclass
class PathsConfig:
    dataset_dir: str = "data/images"
    # Multiple folders: {name: folder}. When not empty it takes precedence over ``dataset_dir``
    # and the image_id becomes "<name>__<file>", because file names repeat across folders.
    dataset_dirs: dict[str, str] = field(default_factory=dict)
    results_dir: str = "active_learning_tool/results"
    embeddings_dir: str = "data/embeddings"
    models_dir: str = "models"
    # JSON {"image_ids": [...]}: if set, the hub works only on these bees (a later phase
    # of a pipeline, e.g. only the well-segmented bees). Empty = every bee with an embedding.
    subset_file: str = ""


@dataclass
class LabelsConfig:
    classes: list[str] = field(default_factory=list)
    label_aliases: dict[str, str] = field(default_factory=dict)
    reject_classes: list[str] = field(default_factory=list)


@dataclass
class ExtractorConfig:
    model: str = "dinov2_vits14"
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 4
    shard_size: int = 2048
    device: str = "auto"

    def __post_init__(self) -> None:
        _check(self.batch_size > 0 and self.shard_size > 0, "extractor.batch_size/shard_size > 0")


@dataclass
class KNNConfig:
    k: int = 10
    temperature: float = 0.07
    backend: str = "auto"
    unassigned_fraction: float = 0.05

    def __post_init__(self) -> None:
        _check(self.k >= 1, "knn.k >= 1")
        _check(self.temperature > 0, "knn.temperature > 0")
        _check(self.backend in ("auto", "faiss", "sklearn", "numpy"), "knn.backend unknown")
        _check(0 <= self.unassigned_fraction < 1, "knn.unassigned_fraction in [0, 1)")


@dataclass
class PropagationConfig:
    enabled: bool = True
    n_neighbors: int = 15
    alpha: float = 0.8
    max_iter: int = 30
    tol: float = 1e-4

    def __post_init__(self) -> None:
        _check(0 < self.alpha < 1, "propagation.alpha in (0, 1)")
        _check(self.n_neighbors >= 1, "propagation.n_neighbors >= 1")


@dataclass
class QueryConfig:
    batch_size: int = 30
    cold_start_threshold: int = 50
    uncertainty: str = "margin"
    uncertainty_source: str = "propagation"
    candidate_multiplier: int = 10
    random_fraction: float = 0.1
    novelty_fraction: float = 0.2

    def __post_init__(self) -> None:
        _check(self.batch_size >= 1, "query.batch_size >= 1")
        _check(self.uncertainty in UNCERTAINTIES, f"query.uncertainty in {UNCERTAINTIES}")
        _check(self.uncertainty_source in ("knn", "propagation"), "query.uncertainty_source in (knn, propagation)")
        _check(self.candidate_multiplier >= 1, "query.candidate_multiplier >= 1")
        _check(0 <= self.random_fraction <= 1, "query.random_fraction in [0, 1]")
        _check(0 <= self.novelty_fraction <= 1, "query.novelty_fraction in [0, 1]")


@dataclass
class ValidationConfig:
    size: int = 100
    seed: int = 123
    # Dataset folders held out to validate the model (e.g. a new batch): the model predicts
    # them but never proposes them, does not learn from them, and their labels stay "holdout".
    holdout_datasets: list[str] = field(default_factory=list)


@dataclass
class HubConfig:
    show_suggestion: bool = False
    assignment_timeout_s: float = 900.0


@dataclass
class RetrainConfig:
    enabled: bool = False
    policy: str = "label_count"
    initial_labels: int = 50
    growth_factor: float = 2.0
    plateau_rounds: int = 3
    metric: str = "macro_f1"
    min_delta: float = 0.0
    trainer: str = "contrastive"
    python: str = "venv/bin/python"
    min_labels: int = 20
    require_improvement: bool = True
    pseudo_label_threshold: float = 0.95
    unfreeze_blocks: int = 2
    steps: int = 300
    batch_size: int = 96
    labeled_fraction: float = 0.5
    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    temperature: float = 0.1
    projection_dim: int = 128
    num_workers: int = 8
    device: str = "auto"

    def __post_init__(self) -> None:
        _check(self.policy in ("label_count", "plateau", "either"), "retrain.policy unknown")
        _check(self.trainer in ("contrastive", "stub"), "retrain.trainer in (contrastive, stub)")
        _check(self.pseudo_label_threshold > 0, "retrain.pseudo_label_threshold > 0")
        _check(0 < self.labeled_fraction < 1, "retrain.labeled_fraction in (0, 1)")
        _check(self.steps >= 1 and self.batch_size >= 4, "retrain.steps >= 1 and batch_size >= 4")
        _check(self.temperature > 0, "retrain.temperature > 0")
        _check(self.metric in ("macro_f1", "accuracy"), "retrain.metric in (macro_f1, accuracy)")
        _check(self.initial_labels >= 1 and self.growth_factor >= 1, "retrain.initial_labels/growth_factor >= 1")


@dataclass
class StoppingConfig:
    """Stopping criteria shown in the hub: they are only a hint and do not stop anything."""

    window: int = 3
    max_pred_change: float = 0.01
    f1_min_delta: float = 0.01

    def __post_init__(self) -> None:
        _check(self.window >= 1, "stopping.window >= 1")
        _check(0 <= self.max_pred_change <= 1, "stopping.max_pred_change in [0, 1]")
        _check(self.f1_min_delta >= 0, "stopping.f1_min_delta >= 0")


@dataclass
class FramesConfig:
    """Link from bee to source frame (see ``bee_al/frames.py``)."""

    # {dataset folder name: path of the extraction manifest}; empty = <cleaned folder>/../raw_bees/
    manifests: dict[str, str] = field(default_factory=dict)
    # regex with the named groups (?P<session>...) and/or (?P<colony>...), searched in the frame name
    frame_pattern: str = ""
    # {session name: folder with the original frames of that session}, also searched in subfolders.
    # Needed when the frame name says nothing (e.g. DSC_5365.JPG): the session is the folder containing it.
    session_dirs: dict[str, str] = field(default_factory=dict)
    # hive assigned to every frame when the regex does not yield one (a single photographed hive)
    colony: str = ""

    def __post_init__(self) -> None:
        if self.frame_pattern:
            try:
                groups = set(re.compile(self.frame_pattern).groupindex)
            except re.error as e:
                raise ValueError(f"Invalid config: frames.frame_pattern is not a regex ({e})") from None
            _check(bool(groups & {"session", "colony"}) and groups <= {"session", "colony"},
                   "frames.frame_pattern must have the named groups 'session' and/or 'colony'")


@dataclass
class AuditConfig:
    """Stratified audit sample (see ``bee_al/audit.py``)."""

    per_stratum: int = 40
    large_stratum: int = 250
    large_threshold: int = 5000
    confidence_bins: list[float] = field(default_factory=lambda: [0.9])
    double_fraction: float = 0.1
    anomaly_split_class: str = ""     # e.g. "Healthy Adult"; empty = no split by anomaly score
    anomaly_quantile: float = 0.9
    seed: int = 7
    # Only the bees of these dataset folders (e.g. the held-out batch); empty = all.
    datasets: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check(self.per_stratum >= 1, "audit.per_stratum >= 1")
        _check(self.large_stratum >= self.per_stratum, "audit.large_stratum >= audit.per_stratum")
        _check(self.large_threshold >= 1, "audit.large_threshold >= 1")
        _check(all(0 < b < 1 for b in self.confidence_bins) and len(set(self.confidence_bins)) == len(self.confidence_bins),
               "audit.confidence_bins: distinct thresholds in (0, 1)")
        _check(0 <= self.double_fraction <= 1, "audit.double_fraction in [0, 1]")
        _check(0 < self.anomaly_quantile < 1, "audit.anomaly_quantile in (0, 1)")


GROUP_BY = ("frame", "session", "colony", "dataset")
LIGHT_RULES = ("disease_detected", "profile_divergent", "unexplained_high")


@dataclass
class ColonyConfig:
    """Colony indicators on the audit-corrected estimates (see ``bee_al/colony.py``)."""

    group_by: str = "session"               # frame | session | colony | dataset
    demographic_classes: list[str] = field(default_factory=list)
    disease_classes: list[str] = field(default_factory=list)
    min_detected: int = 3
    reference_groups: list[str] = field(default_factory=list)
    time_order: dict[str, list[str]] = field(default_factory=dict)  # {colony: [session1, session2, ...]}
    anomaly_threshold: float = 0.0          # 0 = AD indicator disabled
    n_boot: int = 1000
    seed: int = 11
    # Traffic light: yellow if at least one rule fires, red from red_min_rules upwards.
    light_rules: list[str] = field(default_factory=lambda: ["disease_detected", "profile_divergent"])
    red_min_rules: int = 2
    max_unexplained: float = 0.0            # threshold of the unexplained_high rule (0 = off)

    def __post_init__(self) -> None:
        _check(self.group_by in GROUP_BY, f"colony.group_by in {GROUP_BY}")
        _check(self.min_detected >= 1, "colony.min_detected >= 1")
        _check(self.n_boot >= 1, "colony.n_boot >= 1")
        _check(self.anomaly_threshold >= 0, "colony.anomaly_threshold >= 0")
        _check(all(r in LIGHT_RULES for r in self.light_rules), f"colony.light_rules in {LIGHT_RULES}")
        _check(self.red_min_rules >= 1, "colony.red_min_rules >= 1")
        _check(0 <= self.max_unexplained <= 1, "colony.max_unexplained in [0, 1]")


@dataclass
class StageConfig:
    """Pipeline phase this config works on: the hub shows it on every page."""

    name: str = ""          # e.g. "fase1": also ends up in the exported files
    title: str = ""         # highlighted line, e.g. "Phase 1 of 2 - segmentation quality"
    description: str = ""   # what the annotator has to do in this phase


@dataclass
class ExportConfig:
    """Where the clusters go when a phase is closed (see ``bee_al/export.py``)."""

    clusters_dir: str = ""  # one subfolder per export; empty = button disabled
    mode: str = "copy"      # copy | hardlink | symlink

    def __post_init__(self) -> None:
        _check(self.mode in ("copy", "hardlink", "symlink"), "export.mode in (copy, hardlink, symlink)")


@dataclass
class ReviewConfig:
    """Review queue of suspicious labels (see ``bee_al/review.py``)."""

    enabled: bool = True
    k: int = 0                    # 0 = use knn.k
    min_disagreement: float = 0.8 # minimum probability of the suggested class
    min_class_size: int = 5       # below this threshold the class is "rare": separate section
    show_neighbors: int = 6

    def __post_init__(self) -> None:
        _check(self.k >= 0, "review.k >= 0")
        _check(0 < self.min_disagreement <= 1, "review.min_disagreement in (0, 1]")
        _check(self.min_class_size >= 1, "review.min_class_size >= 1")
        _check(self.show_neighbors >= 1, "review.show_neighbors >= 1")


@dataclass
class SimulationConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    max_labels: int = 400
    threshold_metric: str = "macro_f1"
    threshold: float = 0.8
    synthetic_n: int = 3000
    synthetic_dim: int = 64
    synthetic_classes: int = 6
    synthetic_imbalance: float = 0.6
    synthetic_noise: float = 2.0


@dataclass
class ALConfig:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    colony: ColonyConfig = field(default_factory=ColonyConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    root: str = str(REPO_ROOT)  # base of relative paths; not part of the YAML file

    def resolve(self, path: Union[str, Path]) -> Path:
        """Absolute path: relative paths start from the repo root."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else Path(self.root) / p

    @property
    def dataset_sources(self) -> list[tuple[str, Path]]:
        """Image folders as (image_id prefix, folder); empty prefix with a single folder."""
        if self.paths.dataset_dirs:
            return [(f"{name}{ID_SEPARATOR}", self.resolve(d)) for name, d in self.paths.dataset_dirs.items()]
        return [("", self.resolve(self.paths.dataset_dir))]

    @property
    def dataset_dir(self) -> Path:
        """First (or only) dataset folder."""
        return self.dataset_sources[0][1]

    @property
    def results_dir(self) -> Path:
        return self.resolve(self.paths.results_dir)

    @property
    def embeddings_dir(self) -> Path:
        return self.resolve(self.paths.embeddings_dir)

    @property
    def models_dir(self) -> Path:
        return self.resolve(self.paths.models_dir)

    @property
    def clusters_dir(self) -> Optional[Path]:
        return self.resolve(self.export.clusters_dir) if self.export.clusters_dir else None

    @property
    def subset_path(self) -> Optional[Path]:
        return self.resolve(self.paths.subset_file) if self.paths.subset_file else None

    @property
    def al_labels_path(self) -> Path:
        return self.results_dir / "al_labels.csv"

    @property
    def validation_path(self) -> Path:
        return self.results_dir / "validation_set.json"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("root")
        return d

    @property
    def config_id(self) -> str:
        """Short hash of the configuration, saved next to every label."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """Convert ``value`` to the annotated type, with readable errors."""
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        _check(isinstance(value, dict), f"{where} must be a mapping")
        return _from_dict(tp, value, where)
    if origin is list:
        value = [] if value is None else value
        _check(isinstance(value, list), f"{where} must be a list")
        (item_tp,) = typing.get_args(tp)
        return [_coerce(v, item_tp, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        value = {} if value is None else value
        _check(isinstance(value, dict), f"{where} must be a mapping")
        _, value_tp = typing.get_args(tp)
        return {str(k): _coerce(v, value_tp, f"{where}.{k}") for k, v in value.items()}
    if tp is bool:
        _check(isinstance(value, bool), f"{where} must be true/false")
        return value
    if tp is int:
        _check(isinstance(value, int) and not isinstance(value, bool), f"{where} must be an integer")
        return value
    if tp is float:
        _check(isinstance(value, (int, float)) and not isinstance(value, bool), f"{where} must be a number")
        return float(value)
    if tp is str:
        return str(value)
    return value


def _from_dict(cls: type, data: dict[str, Any], where: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    _check(not unknown, f"unknown keys in '{where or 'root'}': {sorted(unknown)}")
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}".lstrip(".")) for k, v in data.items()}
    return cls(**kwargs)


def config_from_dict(data: Optional[dict[str, Any]], root: Union[str, Path, None] = None) -> ALConfig:
    """Build the config from a dict; missing keys take the defaults."""
    data = dict(data or {})
    _check("root" not in data, "'root' does not belong in the configuration file")
    cfg: ALConfig = _from_dict(ALConfig, data)
    for name in cfg.paths.dataset_dirs:
        _check(bool(name) and "/" not in name and ID_SEPARATOR not in name,
               f"paths.dataset_dirs: invalid name {name!r} (no '/' or '{ID_SEPARATOR}' allowed)")
    for name in cfg.validation.holdout_datasets:
        _check(name in cfg.paths.dataset_dirs, f"validation.holdout_datasets: {name!r} is not in paths.dataset_dirs")
    if root is not None:
        cfg.root = str(Path(root).resolve())
    return cfg


def load_config(path: Union[str, Path, None] = None, root: Union[str, Path, None] = None) -> ALConfig:
    """Read the YAML file (default ``config/al_config.yaml``)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return config_from_dict(yaml.safe_load(_expand_env(text)), root=root)


def _expand_env(text: str) -> str:
    """Expand ``${VAR}`` in the YAML text.

    ``BEES_DATA_ROOT`` (the bee dataset) and ``BEE_AL_WORKDIR`` (labels, embeddings
    and models of the active-learning hub) default to ``data/DatasetApi_Ceschi`` and
    ``data/bee_al`` under the repository root when they are not set.
    """
    defaults = {
        "BEES_DATA_ROOT": str(REPO_ROOT.parent / "data" / "DatasetApi_Ceschi"),
        "BEE_AL_WORKDIR": str(REPO_ROOT.parent / "data" / "bee_al"),
    }
    env = {**defaults, **os.environ}
    return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), text)

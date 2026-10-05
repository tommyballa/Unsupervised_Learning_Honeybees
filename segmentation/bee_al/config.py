"""Configurazione tipata, letta da ``config/al_config.yaml``."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "al_config.yaml"

UNCERTAINTIES = ("margin", "entropy", "least_confidence")
ID_SEPARATOR = "__"  # tra nome della cartella e nome del file negli image_id


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Config non valida: {msg}")


@dataclass
class PathsConfig:
    dataset_dir: str = "data/images"
    # Più cartelle: {nome: cartella}. Se non è vuoto ha la precedenza su ``dataset_dir``
    # e l'image_id diventa "<nome>__<file>", perché i nomi dei file si ripetono tra cartelle.
    dataset_dirs: dict[str, str] = field(default_factory=dict)
    results_dir: str = "active_learning_tool/results"
    embeddings_dir: str = "data/embeddings"
    models_dir: str = "models"


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
        _check(self.backend in ("auto", "faiss", "sklearn", "numpy"), "knn.backend sconosciuto")
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
        _check(self.policy in ("label_count", "plateau", "either"), "retrain.policy sconosciuta")
        _check(self.trainer in ("contrastive", "stub"), "retrain.trainer in (contrastive, stub)")
        _check(self.pseudo_label_threshold > 0, "retrain.pseudo_label_threshold > 0")
        _check(0 < self.labeled_fraction < 1, "retrain.labeled_fraction in (0, 1)")
        _check(self.steps >= 1 and self.batch_size >= 4, "retrain.steps >= 1 e batch_size >= 4")
        _check(self.temperature > 0, "retrain.temperature > 0")
        _check(self.metric in ("macro_f1", "accuracy"), "retrain.metric in (macro_f1, accuracy)")
        _check(self.initial_labels >= 1 and self.growth_factor >= 1, "retrain.initial_labels/growth_factor >= 1")


@dataclass
class StoppingConfig:
    """Criteri di stop mostrati nell'hub: sono un'indicazione, non fermano nulla."""

    window: int = 3
    max_pred_change: float = 0.01
    f1_min_delta: float = 0.01

    def __post_init__(self) -> None:
        _check(self.window >= 1, "stopping.window >= 1")
        _check(0 <= self.max_pred_change <= 1, "stopping.max_pred_change in [0, 1]")
        _check(self.f1_min_delta >= 0, "stopping.f1_min_delta >= 0")


@dataclass
class FramesConfig:
    """Collegamento ape -> frame di origine (vedi ``bee_al/frames.py``)."""

    # {nome cartella dataset: percorso del manifest dell'estrazione}; vuoto = <cartella pulita>/../raw_bees/
    manifests: dict[str, str] = field(default_factory=dict)
    # regex con i gruppi nominati (?P<session>...) e/o (?P<colony>...) cercata nel nome del frame
    frame_pattern: str = ""

    def __post_init__(self) -> None:
        if self.frame_pattern:
            try:
                groups = set(re.compile(self.frame_pattern).groupindex)
            except re.error as e:
                raise ValueError(f"Config non valida: frames.frame_pattern non è una regex ({e})") from None
            _check(bool(groups & {"session", "colony"}) and groups <= {"session", "colony"},
                   "frames.frame_pattern deve avere i gruppi nominati 'session' e/o 'colony'")


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
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    root: str = str(REPO_ROOT)  # base dei percorsi relativi; non va nel file YAML

    def resolve(self, path: Union[str, Path]) -> Path:
        """Percorso assoluto: quelli relativi partono dalla root del repo."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else Path(self.root) / p

    @property
    def dataset_sources(self) -> list[tuple[str, Path]]:
        """Cartelle delle immagini come (prefisso dell'image_id, cartella); prefisso vuoto con una sola cartella."""
        if self.paths.dataset_dirs:
            return [(f"{name}{ID_SEPARATOR}", self.resolve(d)) for name, d in self.paths.dataset_dirs.items()]
        return [("", self.resolve(self.paths.dataset_dir))]

    @property
    def dataset_dir(self) -> Path:
        """Prima (o unica) cartella del dataset."""
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
        """Hash breve della configurazione, salvato accanto a ogni etichetta."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """Converte ``value`` nel tipo annotato, con errori leggibili."""
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        _check(isinstance(value, dict), f"{where} deve essere una mappa")
        return _from_dict(tp, value, where)
    if origin is list:
        value = [] if value is None else value
        _check(isinstance(value, list), f"{where} deve essere una lista")
        (item_tp,) = typing.get_args(tp)
        return [_coerce(v, item_tp, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        value = {} if value is None else value
        _check(isinstance(value, dict), f"{where} deve essere una mappa")
        return {str(k): str(v) for k, v in value.items()}
    if tp is bool:
        _check(isinstance(value, bool), f"{where} deve essere true/false")
        return value
    if tp is int:
        _check(isinstance(value, int) and not isinstance(value, bool), f"{where} deve essere un intero")
        return value
    if tp is float:
        _check(isinstance(value, (int, float)) and not isinstance(value, bool), f"{where} deve essere un numero")
        return float(value)
    if tp is str:
        return str(value)
    return value


def _from_dict(cls: type, data: dict[str, Any], where: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    _check(not unknown, f"chiavi sconosciute in '{where or 'root'}': {sorted(unknown)}")
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}".lstrip(".")) for k, v in data.items()}
    return cls(**kwargs)


def config_from_dict(data: Optional[dict[str, Any]], root: Union[str, Path, None] = None) -> ALConfig:
    """Costruisce la config da un dict; le chiavi mancanti prendono i default."""
    data = dict(data or {})
    _check("root" not in data, "'root' non va nel file di configurazione")
    cfg: ALConfig = _from_dict(ALConfig, data)
    for name in cfg.paths.dataset_dirs:
        _check(bool(name) and "/" not in name and ID_SEPARATOR not in name,
               f"paths.dataset_dirs: nome non valido {name!r} (niente '/' né '{ID_SEPARATOR}')")
    if root is not None:
        cfg.root = str(Path(root).resolve())
    return cfg


def load_config(path: Union[str, Path, None] = None, root: Union[str, Path, None] = None) -> ALConfig:
    """Legge il file YAML (default ``config/al_config.yaml``)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f), root=root)

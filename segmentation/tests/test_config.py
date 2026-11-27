import pytest

from bee_al.config import DEFAULT_CONFIG_PATH, config_from_dict, load_config


def test_default_yaml_loads_with_spec_defaults():
    cfg = load_config()
    assert DEFAULT_CONFIG_PATH.exists()
    assert cfg.query.batch_size == 30
    assert cfg.query.cold_start_threshold == 50
    assert cfg.query.candidate_multiplier == 10
    assert cfg.query.random_fraction == pytest.approx(0.1)
    assert cfg.validation.size == 100
    assert cfg.hub.show_suggestion is False
    assert config_from_dict({}).labels.classes == []     # no class hard-coded
    assert cfg.labels.reject_classes == []               # the file only has suggestions for the quick labels


def test_relative_paths_resolve_against_root(tmp_path):
    cfg = config_from_dict({"paths": {"results_dir": "res"}}, root=tmp_path)
    assert cfg.al_labels_path == tmp_path / "res" / "al_labels.csv"
    cfg = config_from_dict({"paths": {"embeddings_dir": "emb", "models_dir": "/absolute/models"}}, root=tmp_path)
    assert cfg.embeddings_dir == tmp_path / "emb" and str(cfg.models_dir) == "/absolute/models"


def test_unknown_key_and_bad_values_are_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        config_from_dict({"query": {"batch_sise": 3}})
    with pytest.raises(ValueError):
        config_from_dict({"query": {"uncertainty": "boh"}})
    with pytest.raises(ValueError):
        config_from_dict({"query": {"batch_size": "30"}})


def test_config_id_changes_with_content():
    a = config_from_dict({})
    b = config_from_dict({"knn": {"k": 5}})
    assert a.config_id == config_from_dict({}).config_id
    assert a.config_id != b.config_id

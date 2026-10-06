from __future__ import annotations

import pytest

from ivcrypto.config import (
    CleaningConfig,
    Config,
    config_from_dict,
    config_to_dict,
    load_config,
)


def test_default_toml_documents_the_code_defaults(repo_root):
    assert load_config(repo_root / "config" / "default.toml") == Config()


def test_no_path_means_defaults():
    assert load_config(None) == Config()


def test_partial_override_keeps_other_defaults(tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text("[cleaning]\nmin_days_to_expiry = 1\n")
    config = load_config(path)
    assert config.cleaning.min_days_to_expiry == 1.0
    assert isinstance(config.cleaning.min_days_to_expiry, float)
    assert config.cleaning.max_relative_spread == CleaningConfig().max_relative_spread


def test_round_trip_through_dict():
    config = config_from_dict({"cleaning": {"forward_method": "underlying", "parity_pairs": 4}})
    assert config_from_dict(config_to_dict(config)) == config


@pytest.mark.parametrize(
    ("data", "error", "match"),
    [
        ({"cleaningg": {}}, ValueError, "unknown config sections"),
        ({"cleaning": {"min_days": 1.0}}, ValueError, "unknown keys"),
        ({"cleaning": {"parity_pairs": 2.5}}, TypeError, "expected int"),
        ({"cleaning": {"forward_method": 3}}, TypeError, "expected str"),
        ({"cleaning": {"parity_pairs": True}}, TypeError, "expected int"),
        ({"cleaning": {"forward_method": "spot"}}, ValueError, "forward_method"),
        ({"cleaning": {"max_relative_spread": 0.0}}, ValueError, "max_relative_spread"),
        ({"cleaning": 5}, TypeError, "must be a table"),
    ],
)
def test_invalid_config_is_rejected(data, error, match):
    with pytest.raises(error, match=match):
        config_from_dict(data)

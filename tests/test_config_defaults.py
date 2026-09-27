from __future__ import annotations

from synthetic_bit_sequence_majority_rule.domain.params import AppConfig

SECTIONS = (
    "project",
    "run",
    "dataset",
    "preprocessing",
    "metrics",
    "neighbors",
    "k_values",
    "majority_rule",
    "binary_sequence",
    "decimal_encoding",
    "statistics",
    "exports",
    "notes",
)


def test_omitted_config_keys_fall_back_to_the_dataclass_defaults() -> None:
    # Every section is present but empty, so every field is read through its default.
    config = AppConfig.from_dict({section: {} for section in SECTIONS})

    assert config == AppConfig()
    assert config.run.run_name == "default_run"
    assert str(config.run.output_root).replace("\\", "/") == "outputs/runs"

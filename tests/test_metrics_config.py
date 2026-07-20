from __future__ import annotations

import pytest

from synthetic_bit_sequence_majority_rule.domain.params import MetricsConfig


def test_metrics_config_accepts_manhattan() -> None:
    cfg = MetricsConfig(enabled=["euclidean", "chebyshev", "canberra", "manhattan"])

    cfg.validate()

    assert cfg.enabled == ["euclidean", "chebyshev", "canberra", "manhattan"]


def test_metrics_config_rejects_unknown_metric() -> None:
    cfg = MetricsConfig(enabled=["euclidean", "unknown"])

    with pytest.raises(ValueError, match="invalid metrics"):
        cfg.validate()

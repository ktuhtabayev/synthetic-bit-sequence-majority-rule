"""Locations inside the repository checkout the package is installed from (editable install)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"
EXPERIMENTS_CONFIG_PATH = CONFIGS_DIR / "experiments.yaml"

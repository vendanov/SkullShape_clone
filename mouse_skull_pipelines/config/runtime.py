from pathlib import Path
from typing import Dict, List, Optional

import yaml

APP_VERSION = "mouse_skull_pipelines"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _get_app_version() -> str:
    return APP_VERSION


def load_yaml_mapping(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping.")
    return data


def _coerce_override_value(value: str, template):
    if isinstance(template, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def apply_config_overrides(config: Dict, overrides: Optional[List[str]]) -> None:
    if not overrides:
        return
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            raise ValueError("Override key cannot be empty.")
        if key in config:
            config[key] = _coerce_override_value(value, config[key])
        else:
            config[key] = value

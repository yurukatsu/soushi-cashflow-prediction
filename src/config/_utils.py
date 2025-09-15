from pathlib import Path

import yaml

from src.typing import AnyDict, StringOrPath


def load_yaml(filepath: StringOrPath) -> AnyDict:
    """
    Load a YAML file from the specified path and return its contents as a dictionary.

    Args:
        filepath (Path):
            The path to the YAML file.

    Returns:
        dict[str, Any]:
            The contents of the YAML file.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.suffix not in {".yml", ".yaml"}:
        raise ValueError(f"Invalid file extension: {filepath.suffix}")

    if not filepath.exists():
        raise FileNotFoundError(f"Yaml file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

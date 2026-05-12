import tomllib
import json
from pathlib import Path
from types import SimpleNamespace


def load_config(config_path: Path) -> SimpleNamespace:
    """
    Load config.toml and return an object with attributes from the config file.
    Paths are resolved relative to the config file's parent directory.
    """
    config_dir = config_path.parent.resolve()
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    def resolve_path(value, base_dir):
        """Resolve a path string relative to the config directory."""
        if isinstance(value, str):
            # Resolve relative paths against config directory
            return (base_dir / value).resolve()
        return value

    def namespace_from_dict(d: dict, base_dir: Path) -> SimpleNamespace:
        """Recursively convert a dict to a SimpleNamespace, resolving paths."""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = namespace_from_dict(v, base_dir)
            elif isinstance(v, list):
                result[k] = [resolve_path(item, base_dir) if isinstance(item, str) else item for item in v]
            elif k.endswith("_file") or k.endswith("_dir")  or k.endswith("_path"):
                result[k] = resolve_path(v, base_dir)
            else:
                result[k] = v
        return SimpleNamespace(**result)

    return namespace_from_dict(data, config_dir)


def read_json(file_path: Path):
    """Read JSON data from a file and return it as a Python object."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(file_path: Path, data):
    """Write JSON data to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

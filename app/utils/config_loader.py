import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    merged = {}
    for config in configs:
        merged.update(config)
    return merged

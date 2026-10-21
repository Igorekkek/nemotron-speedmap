import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from core.fastinfer import ExperimentConfig, DatasetSpec, GenParams
from core.fastinfer.types import todo


class ConfigException(Exception):

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Following problems found: \n"+"\n".join(errors))





def load_yaml(path: str) -> dict[str, Any]:
    pth = Path(path)
    if not pth.exists():
        raise FileNotFoundError("not found: "+path)
    try:
        with open(pth) as data_raw:
            data_yaml = yaml.safe_load(data_raw)
    except yaml.YAMLError as e:
        raise e
    return data_yaml


def validate_config(data: dict[str, Any]):
    errors: list[str] = []

    REQUIRED_KEYS = set("name family hypothesis device mode dataset gen base".split())

    for rkey in REQUIRED_KEYS:
        if not (rkey in data):
            errors.append(f"required key {rkey} not present")

    todo("other checks")

    return errors



def load_experiment_config(path: str) -> ExperimentConfig:
    data_yaml = load_yaml(path)
    errors = validate_config(data_yaml)

    if (len(errors)>0):
        raise ConfigException(errors)
    

    result = ExperimentConfig(
        name = data_yaml.get('name'),
        family = int(data_yaml.get('family')),
        hypothesis = data_yaml.get('hypothesis'),
        device = data_yaml.get('device'),
        mode = data_yaml.get('mode'),
        model = data_yaml.get('model', None),
        quant = data_yaml.get('quant', None),
        dataset = DatasetSpec(data_yaml.get('dataset')),
        gen = GenParams(data_yaml.get('gen')),
        base = data_yaml.get('base'),
        variants = data_yaml.get('variants', []),
        repeats = data_yaml.get('repeats', 3),
        budget_usd = data_yaml.get('budget_usd', 0),
        timeout_min = data_yaml.get('timeout_min', 600),
        notes = data_yaml.get('notes', '')
    )

    return result






def config_hash(conf: ExperimentConfig) -> str:
    """Стабильный отпечаток конфига: SHA-256 от канонического JSON."""
    data = dataclasses.asdict(conf)
    canonical = json.dumps(
        data,
        sort_keys=True,          # order irrelevant
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()









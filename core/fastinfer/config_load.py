import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from fastinfer import ExperimentConfig, DatasetSpec, GenParams, VariantSpec
from fastinfer.shared import todo


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




def load_experiment_config(path: str) -> ExperimentConfig:
    data_yaml = load_yaml(path)
    errors = validate_config(data_yaml)

    if (len(errors)>0):
        raise ConfigException(errors)


    dataset_dict: dict[str, Any] = data_yaml.get('dataset') or dict()
    gen_params_dict: dict[str, Any] = data_yaml.get('gen') or dict()
    base_dict: dict[str, Any] = data_yaml.get('base') or dict()
    variant_list = list(VariantSpec(**i) for i in data_yaml.get('variants', []))

    result = ExperimentConfig(
        name = data_yaml.get('name'),
        family = int(data_yaml.get('family')),
        hypothesis = data_yaml.get('hypothesis'),
        device = data_yaml.get('device'),
        mode = data_yaml.get('mode'),
        model = data_yaml.get('model', None),
        quant = data_yaml.get('quant', None),
        dataset = DatasetSpec(**dataset_dict),
        gen = GenParams(**gen_params_dict),
        base = VariantSpec(**base_dict),
        variants = variant_list,
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



if True or "Global constants":
    DEVICES = frozenset({"kaggle-t4x2", "laptop-780m", "mac", "pc-cpu", "any"})
    MODES = frozenset({"speed", "quality", "offline", "analysis"})
    ENGINES = frozenset({"llamacpp", "hf", "api"})
    DATASETS = frozenset({"spec_bench", "gsm8k", "math500", "aime"})
    REASONING = frozenset({"off", "on"})
    FA_VALUES = frozenset({"on", "off", "auto"})

    # Флаги llama.cpp — docs/core_api.md, раздел 3.
    FLAG_KEYS = frozenset({
        "ngl", "fa", "sm", "ts", "t", "tb", "ub", "ctx", "np",
        "ctk", "ctv", "spec_type", "model_draft", "ngld",
        "spec_draft_n_max", "spec_draft_n_min", "spec_draft_p_min",
        "spec_draft_device", "reasoning_budget", "override_kv", "n_cpu_moe", "extra",
    })
    INT_FLAGS = frozenset({
        "ngl", "t", "tb", "ub", "ctx", "np", "ngld",
        "spec_draft_n_max", "spec_draft_n_min", "reasoning_budget", "n_cpu_moe",
    })
    FLAG_MIN = {
        "ngl": 0, "t": 0, "tb": 0, "ub": 1, "ctx": 1, "np": 1, "ngld": 0,
        "spec_draft_n_max": 0, "spec_draft_n_min": 0, "reasoning_budget": -1, "n_cpu_moe": 0,
    }
    GEN_FIELDS = frozenset(field.name for field in dataclasses.fields(GenParams))


def validate_config(data: dict[str, Any]) -> list[str]:
    """Проверить config.yaml целиком: вернуть все ошибки сразу (пустой список — конфиг годен).

    Проверяются формат, обязательные поля и допустимые значения — без загрузки моделей и чтения
    файлов (docs/core_api.md, разделы 3 и 7.2). Ошибки не бросаются, чтобы показать их все.
    """
    errors: list[str] = []

    # FakeEngine и локальные промпты — для тестовых сценариев без моделей и сети.
    allowed_engines = ENGINES | {"fake"}
    allowed_datasets = DATASETS | {"local"}
    fake_flags = frozenset({"fake_speed", "fake_seed"})

    def add(where: str, message: str) -> None:
        errors.append(f"{where}: {message}")

    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def check_string(value: Any, where: str) -> None:
        if not isinstance(value, str) or not value.strip():
            add(where, f"expected non-empty str, got {type(value).__name__}")

    if not isinstance(data, dict):
        return [f"config: expected dict, got {type(data).__name__}"]

    def check_gen_value(field: str, value: Any, where: str) -> None:
        if isinstance(value, bool):
            add(where, "bool is not allowed")
        elif field == "max_tokens":
            if not (is_int(value) and value > 0):
                add(where, "expected int > 0")
        elif field == "temperature":
            if not (is_number(value) and 0 <= value <= 2):
                add(where, "expected number in [0, 2]")
        elif field == "top_p":
            if not (is_number(value) and 0 <= value <= 1):
                add(where, "expected number in [0, 1]")
        elif field == "seed":
            if not is_int(value):
                add(where, "expected int")
        elif field == "reasoning":
            if not isinstance(value, str) or value not in REASONING:
                add(where, f"expected one of {sorted(REASONING)}")
        elif field == "reasoning_budget":
            if not (is_int(value) and value >= -1):
                add(where, "expected int >= -1")
        elif field == "logprobs":
            if not (is_int(value) and 0 <= value <= 20):
                add(where, "expected int in 0..20")
        elif field == "stop":
            if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
                add(where, "expected list[str]")

    def check_gen(gen: Any, where: str) -> None:
        if not isinstance(gen, dict):
            add(where, f"expected dict, got {type(gen).__name__}")
            return
        for field, value in gen.items():
            if field not in GEN_FIELDS:
                add(f"{where}.{field}", "unknown GenParams field")
            else:
                check_gen_value(field, value, f"{where}.{field}")

    def check_flag_value(key: str, value: Any, where: str) -> None:
        if isinstance(value, bool):
            add(where, "bool is not allowed; write on/off in quotes")
        elif key in INT_FLAGS:
            if not is_int(value):
                add(where, f"expected int, got {type(value).__name__}")
            elif value < FLAG_MIN[key]:
                add(where, f"must be >= {FLAG_MIN[key]}")
        elif key == "spec_draft_p_min":
            if not (is_number(value) and 0 <= value <= 1):
                add(where, "expected number in [0, 1]")
        elif key == "fa":
            if not isinstance(value, str) or value not in FA_VALUES:
                add(where, f"expected one of {sorted(FA_VALUES)}")
        elif key == "extra":
            if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
                add(where, "expected list[str]")
        elif key == "override_kv":
            valid = isinstance(value, str) or (
                    isinstance(value, list) and all(isinstance(item, str) for item in value)
            )
            if not valid:
                add(where, "expected str or list[str]")
        elif not isinstance(value, str) or not value.strip():
            add(where, f"expected non-empty str, got {type(value).__name__}")

    def check_flags(flags: Any, where: str, engine: str = "llamacpp") -> None:
        if not isinstance(flags, dict):
            add(where, f"expected dict, got {type(flags).__name__}")
            return
        for key, value in flags.items():
            if key in fake_flags:
                if engine != "fake":
                    add(f"{where}.{key}", "unknown llama.cpp flag")
                elif key == "fake_speed":
                    if not (is_number(value) and value > 0):
                        add(f"{where}.{key}", "expected number > 0")
                elif not is_int(value):
                    add(f"{where}.{key}", "expected int")
            elif key not in FLAG_KEYS:
                add(f"{where}.{key}", "unknown llama.cpp flag")
            else:
                check_flag_value(key, value, f"{where}.{key}")

    def check_sweep(sweep: Any, where: str) -> None:
        if not isinstance(sweep, dict):
            add(where, f"expected dict, got {type(sweep).__name__}")
            return
        if not sweep:
            add(where, "must not be empty")
            return
        for key, values in sweep.items():
            if not isinstance(values, list) or not values:
                add(f"{where}.{key}", "expected non-empty list of values")
                continue
            if key in {"model", "quant"}:
                for value in values:
                    if not isinstance(value, str) or not value.strip():
                        add(f"{where}.{key}", f"bad {key}: {value!r}")
            elif key.startswith("gen."):
                field = key[len("gen."):]
                if field not in GEN_FIELDS:
                    add(f"{where}.{key}", "unknown GenParams field")
                else:
                    for value in values:
                        check_gen_value(field, value, f"{where}.{key}")
            elif key in FLAG_KEYS:
                for value in values:
                    check_flag_value(key, value, f"{where}.{key}")
            else:
                add(f"{where}.{key}", "unknown sweep key")

    def check_variant(spec: Any, where: str, require_name: bool) -> None:
        if not isinstance(spec, dict):
            add(where, f"expected dict, got {type(spec).__name__}")
            return
        allowed = {"name", "engine", "model", "quant", "flags", "gen", "analysis", "meta", "sweep"}
        for key in spec:
            if key not in allowed:
                add(f"{where}.{key}", "unknown VariantSpec field")
        if require_name:
            name = spec.get("name")
            if not isinstance(name, str) or not name.strip():
                add(f"{where}.name", "required non-empty str")
        engine = spec.get("engine", "llamacpp")
        if not isinstance(engine, str) or engine not in allowed_engines:
            add(f"{where}.engine", f"expected one of {sorted(allowed_engines)}, got {engine!r}")
        for key in ("model", "quant"):
            if key in spec:
                check_string(spec[key], f"{where}.{key}")
        if "flags" in spec:
            check_flags(spec["flags"], f"{where}.flags", engine)
        if "gen" in spec:
            check_gen(spec["gen"], f"{where}.gen")
        for key in ("analysis", "meta", "sweep"):
            if key in spec and not isinstance(spec[key], dict):
                add(f"{where}.{key}", f"expected dict, got {type(spec[key]).__name__}")
        if isinstance(spec.get("sweep"), dict):
            check_sweep(spec["sweep"], f"{where}.sweep")

    def check_dataset(dataset: Any, where: str) -> None:
        if not isinstance(dataset, dict):
            add(where, f"expected dict, got {type(dataset).__name__}")
            return
        for key in dataset:
            if key not in {"name", "n", "seed", "tasks"}:
                add(f"{where}.{key}", "unknown dataset field")
        name = dataset.get("name")
        if not isinstance(name, str) or name not in allowed_datasets:
            add(f"{where}.name", f"expected one of {sorted(allowed_datasets)}, got {name!r}")
        if "n" in dataset:
            n = dataset["n"]
            if n is not None and not (is_int(n) and n > 0):
                add(f"{where}.n", "expected int > 0 or null")
        if "seed" in dataset and not is_int(dataset["seed"]):
            add(f"{where}.seed", "expected int")
        if "tasks" in dataset:
            tasks = dataset["tasks"]
            if tasks is not None and not (
                    isinstance(tasks, list) and all(isinstance(task, str) for task in tasks)
            ):
                add(f"{where}.tasks", "expected list[str] or null")

    required = ["name", "family", "hypothesis", "device", "mode", "dataset", "base", "variants"]
    for key in required:
        if key not in data:
            add("config", f"required key {key!r} not present")

    allowed_top = set(required) | {"model", "quant", "gen", "repeats", "budget_usd", "timeout_min", "notes"}
    for key in data:
        if key not in allowed_top:
            add(str(key), "unknown config field")

    if "name" in data:
        check_string(data["name"], "name")

    if "family" in data:
        family = data["family"]
        if not is_int(family) or not 1 <= family <= 12:
            add("family", "expected int in 1..12")

    if "hypothesis" in data:
        check_string(data["hypothesis"], "hypothesis")

    if "device" in data:
        device = data["device"]
        if not isinstance(device, str) or device not in DEVICES:
            add("device", f"expected one of {sorted(DEVICES)}, got {device!r}")

    if "mode" in data:
        mode = data["mode"]
        if not isinstance(mode, str) or mode not in MODES:
            add("mode", f"expected one of {sorted(MODES)}, got {mode!r}")

    for key in ("model", "quant"):
        if key in data:
            check_string(data[key], key)

    if "gen" in data:
        check_gen(data["gen"], "gen")

    budget: float | None
    if "budget_usd" in data:
        budget_value = data["budget_usd"]
        if not (is_number(budget_value) and budget_value >= 0):
            add("budget_usd", "expected number >= 0")
            budget = None
        else:
            budget = float(budget_value)
    else:
        budget = 0.0

    if "repeats" in data:
        repeats = data["repeats"]
        if not (is_int(repeats) and repeats >= 1):
            add("repeats", "expected int >= 1")
        elif data.get("mode") == "quality" and repeats != 1:
            add("repeats", "mode=quality always uses 1 repeat")

    if "timeout_min" in data:
        timeout = data["timeout_min"]
        if not (is_int(timeout) and timeout > 0):
            add("timeout_min", "expected int > 0")

    if "notes" in data and not isinstance(data["notes"], str):
        add("notes", f"expected str, got {type(data['notes']).__name__}")

    if "dataset" in data:
        check_dataset(data["dataset"], "dataset")

    base = data.get("base")
    if "base" in data:
        check_variant(base, "base", require_name=False)

    variants = data.get("variants")
    if "variants" in data:
        if not isinstance(variants, list):
            add("variants", f"expected list, got {type(variants).__name__}")
            variants = []
        else:
            if not variants and data.get("mode") != "analysis":
                add("variants", "empty list is allowed only in mode=analysis")
            seen: set[str] = set()
            for index, variant in enumerate(variants):
                where = f"variants[{index}]"
                check_variant(variant, where, require_name=True)
                if isinstance(variant, dict):
                    variant_name = variant.get("name")
                    if isinstance(variant_name, str) and variant_name:
                        if variant_name in seen:
                            add(f"{where}.name", f"duplicate variant name {variant_name!r}")
                        seen.add(variant_name)
    else:
        variants = []

    base_model = base.get("model") if isinstance(base, dict) else None
    default_model = data.get("model") or base_model
    if isinstance(base, dict) and not default_model:
        add("base.model", "model is not set at top level or in base")
    if isinstance(variants, list):
        for index, variant in enumerate(variants):
            if isinstance(variant, dict) and not (variant.get("model") or default_model):
                add(f"variants[{index}].model", "model is not set here, in base or at top level")

    base_engine = base.get("engine", "llamacpp") if isinstance(base, dict) else "llamacpp"
    if not isinstance(base_engine, str):
        base_engine = "llamacpp"
    used_engines = {base_engine}
    if isinstance(variants, list):
        for variant in variants:
            if isinstance(variant, dict):
                engine = variant.get("engine", base_engine)
                if isinstance(engine, str):
                    used_engines.add(engine)
    if "api" in used_engines and budget is not None and budget <= 0:
        add("budget_usd", "must be > 0 when an engine is api; 0 forbids API calls")

    if data.get("mode") == "analysis":
        has_analysis = isinstance(base, dict) and isinstance(base.get("analysis"), dict) and bool(base["analysis"])
        if isinstance(variants, list):
            has_analysis = has_analysis or any(
                isinstance(variant, dict)
                and isinstance(variant.get("analysis"), dict)
                and bool(variant["analysis"])
                for variant in variants
            )
        if not has_analysis:
            add("base.analysis", "mode=analysis requires non-empty analysis settings")

    return errors




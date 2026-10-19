"""Тесты контракта ядра: всё, что описано в docs/core_api.md (разделы 3–7.1), есть на месте.

Тесты проходят уже сейчас, пока ядро не реализовано. Если тест упал после правки типов, интерфейса движка
или сигнатур — это изменение контракта: оно идёт только через PR, который смотрят все трое (раздел 14).
"""

import dataclasses
import inspect
import json
import types
import typing
from pathlib import Path

import pytest
import yaml

import fastinfer
from fastinfer import types as contract_types
from fastinfer.engines import Engine

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = sorted((REPO / "core" / "templates").glob("*.yaml"))
RESULT_EXAMPLE = REPO / "docs" / "result_example.json"

# Типы раздела 5 и их поля — по порядку.
TYPE_FIELDS = {
    "Prompt": ["id", "task", "turns", "reference", "meta"],
    "GenParams": ["max_tokens", "temperature", "top_p", "seed", "reasoning", "reasoning_budget", "logprobs", "stop"],
    "Generation": ["prompt_id", "turn", "text", "token_ids", "n_prompt", "n_gen", "n_thinking", "n_answer",
                   "ttft_ms", "decode_ms", "total_ms", "tok_s", "draft_n", "draft_accepted", "logprobs",
                   "finish_reason", "cost_usd"],
    "SpeedStats": ["tok_s_median", "tok_s_p10", "tok_s_p90", "ttft_ms_median", "total_s_median", "acceptance",
                   "tokens_per_pass", "frac_of_ceiling"],
    "QualityStats": ["accuracy", "ci95", "n", "mean_thinking_tokens", "mean_answer_tokens", "cost_usd"],
    "Report": ["variant", "mode", "generations", "speed", "quality", "engine_info"],
    "Divergence": ["prompt_id", "position", "base_margin"],
    "Comparison": ["variant", "base", "speedup", "speedup_ci95", "total_time_ratio", "exact_match", "divergences",
                   "numeric_divergences", "accuracy_delta", "accuracy_delta_ci95", "tokens_ratio", "cost_ratio",
                   "verdict", "warnings"],
    "AcceptStats": ["alpha", "tau_by_gamma", "accepted_len_hist", "predicted_speedup_by_gamma", "n_tokens"],
    "DeviceInfo": ["kind", "backend", "gpus", "cpu", "cores", "ram_gb", "os", "bandwidth_gbs", "versions"],
    "ModelSpec": ["name", "kind", "hf_repo", "gguf_repo", "gguf_files", "arch", "params_per_token", "sampling",
                  "reasoning_switch", "api_id", "license"],
    "DatasetSpec": ["name", "n", "seed", "tasks"],
    "VariantSpec": ["name", "engine", "model", "quant", "flags", "gen", "analysis", "meta", "sweep"],
    "ExperimentConfig": ["name", "family", "hypothesis", "device", "mode", "model", "quant", "dataset", "gen",
                         "base", "variants", "repeats", "budget_usd", "timeout_min", "notes"],
    "RunResult": ["schema", "job_id", "commit", "name", "family", "mode", "device", "started_at", "finished_at",
                  "ceiling_tok_s", "variants", "comparison", "accept", "analysis", "cost_usd", "warnings"],
}

# Методы движка (раздел 6) и их параметры без self.
ENGINE_METHODS = {
    "start": [],
    "stop": [],
    "generate": ["prompt", "gen"],
    "generate_many": ["prompts", "gen", "parallel"],
    "tokenize": ["text"],
    "info": [],
    "__enter__": [],
    "__exit__": ["exc_type", "exc", "tb"],
}

# Функции для исследователя (раздел 7.1) и их параметры.
FUNCTIONS = {
    "load_config": ["path"],
    "run_experiment": ["cfg", "out_dir"],
    "load_model": ["name", "engine", "quant", "flags"],
    "load_prompts": ["dataset", "n", "seed", "tasks"],
    "generate": ["engine", "prompts", "gen", "parallel"],
    "measure": ["engine", "prompts", "gen", "mode", "repeats"],
    "compare": ["base", "variant"],
    "save_results": ["result", "out_dir"],
    "device_info": [],
    "ceiling": ["model", "quant", "device"],
}

# Ещё не реализованные функции. Реализовали функцию — уберите её отсюда.
NOT_IMPLEMENTED = set(FUNCTIONS)

# Формат config.yaml (раздел 3).
REQUIRED_CONFIG_FIELDS = ["name", "family", "hypothesis", "device", "mode", "dataset", "base", "variants"]
DEVICES = {"kaggle-t4x2", "laptop-780m", "mac", "pc-cpu", "any"}
MODES = {"speed", "quality", "offline", "analysis"}
ENGINES = {"llamacpp", "hf", "api"}


def empty_value(tp: typing.Any) -> typing.Any:
    """Пустое значение типа: "" для str, 0 для int, [] для list, None для «X | None»."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        return None
    if dataclasses.is_dataclass(tp):
        return make_empty(tp)
    if origin is tuple:
        return tuple(empty_value(arg) for arg in typing.get_args(tp))
    return (origin or tp)()


def make_empty(cls: type) -> typing.Any:
    """Создать dataclass: поля со значением по умолчанию — как есть, в обязательные — пустые значения."""
    hints = typing.get_type_hints(cls)
    required = [f for f in dataclasses.fields(cls)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    return cls(**{f.name: empty_value(hints[f.name]) for f in required})


# --- пакет ---

def test_package_exports_contract():
    for name in [*FUNCTIONS, "Engine", *TYPE_FIELDS]:
        assert hasattr(fastinfer, name), name
        assert name in fastinfer.__all__, name


# --- типы (раздел 5) ---

def test_no_unexpected_types():
    found = {name for name, obj in vars(contract_types).items()
             if dataclasses.is_dataclass(obj) and obj.__module__ == contract_types.__name__}
    assert found == set(TYPE_FIELDS)


@pytest.mark.parametrize("name", TYPE_FIELDS)
def test_type_fields(name):
    cls = getattr(contract_types, name)
    assert dataclasses.is_dataclass(cls)
    assert [f.name for f in dataclasses.fields(cls)] == TYPE_FIELDS[name]


@pytest.mark.parametrize("name", TYPE_FIELDS)
def test_type_can_be_created(name):
    obj = make_empty(getattr(contract_types, name))
    assert dataclasses.asdict(obj).keys() == set(TYPE_FIELDS[name])


def test_gen_params_defaults_are_greedy_without_reasoning():
    gen = fastinfer.GenParams()
    assert (gen.max_tokens, gen.temperature, gen.reasoning, gen.reasoning_budget) == (256, 0.0, "off", -1)


# --- интерфейс движка (раздел 6) ---

def test_engine_is_protocol():
    assert typing.Protocol in Engine.__mro__
    assert Engine.__annotations__.get("name") is str


@pytest.mark.parametrize("method", ENGINE_METHODS)
def test_engine_method(method):
    params = list(inspect.signature(getattr(Engine, method)).parameters)
    assert params == ["self", *ENGINE_METHODS[method]]


# --- функции для исследователя (раздел 7.1) ---

@pytest.mark.parametrize("name", FUNCTIONS)
def test_function_signature(name):
    fn = getattr(fastinfer, name)
    sig = inspect.signature(fn)
    assert list(sig.parameters) == FUNCTIONS[name]
    untyped = [p.name for p in sig.parameters.values() if p.annotation is inspect.Parameter.empty]
    assert not untyped, f"параметры без типов: {untyped}"
    assert sig.return_annotation is not inspect.Signature.empty, "нет типа результата"
    assert fn.__doc__, "нет docstring"


@pytest.mark.parametrize("name", sorted(NOT_IMPLEMENTED))
def test_function_not_implemented_yet(name):
    fn = getattr(fastinfer, name)
    required = [p for p in inspect.signature(fn).parameters.values() if p.default is inspect.Parameter.empty]
    with pytest.raises(NotImplementedError):
        fn(*[None] * len(required))


# --- шаблоны config.yaml (раздел 3) ---

def test_templates_exist():
    assert TEMPLATES, "в core/templates нет ни одного шаблона"


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_template(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    missing = [f for f in REQUIRED_CONFIG_FIELDS if f not in cfg]
    assert not missing, f"нет обязательных полей: {missing}"
    assert "model" in cfg or "model" in cfg["base"], "модель не задана ни сверху, ни в base"

    assert isinstance(cfg["family"], int) and 1 <= cfg["family"] <= 12
    assert cfg["device"] in DEVICES
    assert cfg["mode"] in MODES
    assert "name" in cfg["dataset"]

    assert cfg["base"]["engine"] in ENGINES
    assert isinstance(cfg["variants"], list)
    assert cfg["variants"] or cfg["mode"] == "analysis", "пустой список вариантов бывает только в analysis"
    for variant in cfg["variants"]:
        assert "name" in variant
        assert variant.get("engine", "llamacpp") in ENGINES

    for spec in [cfg["base"], *cfg["variants"]]:
        bools = [key for key, value in spec.get("flags", {}).items() if isinstance(value, bool)]
        assert not bools, f"on и off без кавычек PyYAML читает как true и false: {bools}"


# --- пример result.json (раздел 4) ---

def test_result_example():
    data = json.loads(RESULT_EXAMPLE.read_text(encoding="utf-8"))
    for key in ["schema", "job_id", "commit", "variants", "comparison"]:
        assert key in data, key
    missing = set(TYPE_FIELDS["RunResult"]) - set(data)
    assert not missing, f"в примере нет полей RunResult: {missing}"
    assert set(data["device"]) == set(TYPE_FIELDS["DeviceInfo"])

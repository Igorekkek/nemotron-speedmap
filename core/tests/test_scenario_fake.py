#!/usr/bin/env python3
"""Вертикальный срез этапа 0 на FakeEngine.

Сценарий ровно из шага 1:
    конфиг -> база + вариант (без sweep) -> FakeEngine -> генерации ->
    скорость -> папка результата.

Конфиг лежит в core/resources/test_scenario_fake.yaml.

Запуск как функции из main_tests:
    from core.tests.test_scenario_fake import test_fake_scenario
    test_fake_scenario()

Запуск вручную из корня репозитория:
    python core/tests/test_scenario_fake.py --out /tmp/fake-smoke
    python core/tests/test_scenario_fake.py --config my.yaml --out /tmp/fake-smoke

По умолчанию используется YAML из resources: 4 промпта, 2 повтора,
база и один вариант с другой скоростью. Сеть не нужна, реальные модели не
загружаются.

Файл намеренно самодостаточный: он не меняет пакет fastinfer и не требует
реализации fastinfer.run. Имена внутренних функций/модулей здесь не важны —
важен сам сквозной путь и формат результата.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 1. Импорт пакета в одном стиле: сначала установленный fastinfer, потом локальный core/.
# ---------------------------------------------------------------------------

def _bootstrap_fastinfer() -> None:
    try:
        import fastinfer  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in (here.parent, *here.parents):
        candidates.extend([
            parent / "core",
            parent / "nemotron-speedmap" / "core",
        ])

    for candidate in candidates:
        if (candidate / "fastinfer" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            return

    raise SystemExit(
        "Не найден пакет fastinfer. Установи ядро: pip install -e core, "
        "или запусти файл из корня репозитория nemotron-speedmap."
    )


_bootstrap_fastinfer()

from fastinfer import DeviceInfo, Generation, GenParams, Prompt  # noqa: E402
from fastinfer.engines.fake import FakeEngineV1  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Встроенный конфиг и промпты: срез должен работать без сети и без файлов.
# ---------------------------------------------------------------------------

BUILTIN_PROMPTS: list[Prompt] = [
    Prompt(
        id="fake/qa/1",
        task="qa",
        turns=["Почему генерация языковой модели по одному токену упирается в память?"],
    ),
    Prompt(
        id="fake/summarization/2",
        task="summarization",
        turns=["Сократи до одного предложения: генерация упирается в чтение весов на каждом токене."],
    ),
    Prompt(
        id="fake/rag/3",
        task="rag",
        turns=["По фрагменту документа answer: сколько байт читается на токен у 9B в Q4?"],
    ),
    Prompt(
        id="fake/code/4",
        task="code",
        turns=["Напиши функцию Python, которая считает медиану списка чисел."],
    ),
]

RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"
DEFAULT_CONFIG_PATH = RESOURCES_DIR / "test_scenario_fake.yaml"


# ---------------------------------------------------------------------------
# 3. Ошибки, типы и разбор конфига.
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Ошибка конфига: показывается до запуска движка."""


class SelfCheckError(RuntimeError):
    """Сценарий отработал, но не выполнил условия шага 1."""


@dataclass
class Spec:
    """Разрешённый вариант: база или один из variants."""

    name: str
    engine: str
    model: str
    quant: str | None
    flags: dict[str, Any]
    gen: GenParams
    meta: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine": self.engine,
            "model": self.model,
            "quant": self.quant,
            "flags": dict(self.flags),
            "gen": asdict(self.gen),
            "meta": dict(self.meta),
        }


@dataclass
class Record:
    """Одна генерация одного хода одного промпта."""

    variant: str
    repeat: int
    prompt: Prompt
    generation: Generation


@dataclass
class RunArtifacts:
    result: dict[str, Any]
    out_dir: Path
    base_name: str
    n_prompts: int
    n_variants: int
    repeats: int
    elapsed_s: float


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивно слить словари; override выигрывает."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_config(path: str | Path | None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"конфиг не найден: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{config_path}:{exc.lineno}: {exc.msg}") from exc
    else:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ConfigError("для YAML нужен pyyaml: pip install pyyaml") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"{config_path}: ожидался dict, получено {type(data).__name__}")
    return data


def _as_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: ожидалось целое, получено {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where}: должно быть >= {minimum}, получено {value}")
    return value


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: ожидалось число, получено {value!r}")
    return float(value)


def _make_gen(data: Any) -> GenParams:
    if not isinstance(data, dict):
        raise ConfigError(f"gen: ожидался dict, получено {type(data).__name__}")
    try:
        gen = GenParams(**data)
    except TypeError as exc:
        raise ConfigError(f"gen: {exc}") from exc
    if gen.max_tokens <= 0:
        raise ConfigError("gen.max_tokens: должно быть > 0")
    if gen.temperature < 0:
        raise ConfigError("gen.temperature: должно быть >= 0")
    if gen.logprobs < 0 or gen.logprobs > 20:
        raise ConfigError("gen.logprobs: должно быть в 0..20")
    if gen.reasoning not in {"off", "on"}:
        raise ConfigError("gen.reasoning: ожидалось 'off' или 'on'")
    return gen


def _make_spec(raw: dict[str, Any], inherited: dict[str, Any], default_name: str) -> Spec:
    if not isinstance(raw, dict):
        raise ConfigError(f"{default_name}: ожидался dict, получено {type(raw).__name__}")
    if raw.get("sweep"):
        raise ConfigError(
            f"{default_name}: sweep пока не поддержан в шаге 1; "
            "разверни варианты вручную или дождись шага 2"
        )

    merged = _deep_merge(inherited, raw)
    name = str(raw.get("name") or merged.get("name") or default_name)
    engine = str(raw.get("engine") or merged.get("engine") or "fake").lower()
    if engine != "fake":
        raise ConfigError(
            f"{name}: в шаге 1 поддерживается только engine=fake, получено {engine!r}. "
            "llama.cpp — следующий шаг."
        )

    model = str(raw.get("model") or merged.get("model") or "fake-v1")
    quant = merged.get("quant")
    if quant is not None:
        quant = str(quant)

    flags = merged.get("flags") or {}
    if not isinstance(flags, dict):
        raise ConfigError(f"{name}.flags: ожидался dict")
    unknown = set(flags) - {"fake_speed", "fake_seed"}
    if unknown:
        raise ConfigError(
            f"{name}.flags: неизвестные для FakeEngine ключи {sorted(unknown)}; "
            "поддержаны только fake_speed и fake_seed"
        )
    if "fake_speed" in flags:
        speed = _as_float(flags["fake_speed"], f"{name}.flags.fake_speed")
        if speed <= 0:
            raise ConfigError(f"{name}.flags.fake_speed: должно быть > 0")
        flags["fake_speed"] = speed
    if "fake_seed" in flags:
        flags["fake_seed"] = _as_int(flags["fake_seed"], f"{name}.flags.fake_seed")

    meta = merged.get("meta") or {}
    if not isinstance(meta, dict):
        raise ConfigError(f"{name}.meta: ожидался dict")

    gen_data = merged.get("gen") or {}
    return Spec(
        name=name,
        engine=engine,
        model=model,
        quant=quant,
        flags=dict(flags),
        gen=_make_gen(gen_data),
        meta=dict(meta),
    )


def load_and_resolve_config(path: str | Path | None) -> tuple[dict[str, Any], Spec, list[Spec], list[Prompt]]:
    """Прочитать YAML/JSON, разрешить base -> variant и подготовить промпты."""
    raw = _read_config(path)

    name = str(raw.get("name") or "fake-smoke")
    family = _as_int(raw.get("family", 1), "family", minimum=1)
    if family > 12:
        raise ConfigError("family: должно быть в 1..12")
    hypothesis = str(raw.get("hypothesis") or "")
    device = str(raw.get("device") or "any")
    mode = str(raw.get("mode") or "speed")
    if mode != "speed":
        raise ConfigError(f"mode: в шаге 1 поддерживается только speed, получено {mode!r}")

    base_raw = raw.get("base")
    if not isinstance(base_raw, dict):
        raise ConfigError("base: обязателен и должен быть dict")
    variants_raw = raw.get("variants")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise ConfigError("variants: нужен непустой список (база + хотя бы один вариант)")

    repeats = _as_int(raw.get("repeats", 2), "repeats", minimum=1)
    budget_usd = _as_float(raw.get("budget_usd", 0.0), "budget_usd")
    if budget_usd < 0:
        raise ConfigError("budget_usd: должно быть >= 0")
    timeout_min = _as_int(raw.get("timeout_min", 5), "timeout_min", minimum=1)
    notes = str(raw.get("notes") or "")

    dataset_raw = raw.get("dataset") or {}
    if not isinstance(dataset_raw, dict):
        raise ConfigError("dataset: ожидался dict")
    dataset_name = str(dataset_raw.get("name") or "local")
    dataset_n = dataset_raw.get("n", len(BUILTIN_PROMPTS))
    if dataset_n is None:
        dataset_n = len(BUILTIN_PROMPTS)
    dataset_n = _as_int(dataset_n, "dataset.n", minimum=1)
    if dataset_n > len(BUILTIN_PROMPTS):
        raise ConfigError(
            f"dataset.n: в срезе только {len(BUILTIN_PROMPTS)} встроенных промпта, "
            f"запрошено {dataset_n}"
        )
    dataset_seed = _as_int(dataset_raw.get("seed", 0), "dataset.seed")
    dataset_tasks = dataset_raw.get("tasks")
    if dataset_tasks is not None:
        if not isinstance(dataset_tasks, list) or not all(isinstance(t, str) for t in dataset_tasks):
            raise ConfigError("dataset.tasks: ожидался список строк или null")
        dataset_tasks = [t.strip().lower() for t in dataset_tasks if t.strip()]

    common: dict[str, Any] = {}
    for key in ("model", "quant", "gen", "flags"):
        if key in raw:
            common[key] = raw[key]

    base = _make_spec(base_raw, common, "base")
    variants: list[Spec] = []
    for index, variant_raw in enumerate(variants_raw):
        if not isinstance(variant_raw, dict):
            raise ConfigError(f"variants[{index}]: ожидался dict")
        if not variant_raw.get("name"):
            raise ConfigError(f"variants[{index}].name: обязателен")
        variants.append(_make_spec(variant_raw, base.public(), f"variant-{index + 1}"))

    all_names = [base.name, *(variant.name for variant in variants)]
    if len(set(all_names)) != len(all_names):
        raise ConfigError(f"имена вариантов должны быть уникальны: {all_names}")

    dataset = {
        "name": dataset_name,
        "n": dataset_n,
        "seed": dataset_seed,
        "tasks": dataset_tasks,
    }
    prompts = _load_builtin_prompts(dataset)

    resolved = {
        "name": name,
        "family": family,
        "hypothesis": hypothesis,
        "device": device,
        "mode": mode,
        "model": base.model,
        "quant": base.quant,
        "dataset": dataset,
        "gen": asdict(base.gen),
        "base": base.public(),
        "variants": [variant.public() for variant in variants],
        "repeats": repeats,
        "budget_usd": budget_usd,
        "timeout_min": timeout_min,
        "notes": notes,
    }
    return resolved, base, variants, prompts


def _load_builtin_prompts(dataset: dict[str, Any]) -> list[Prompt]:
    prompts = list(BUILTIN_PROMPTS)
    tasks = dataset.get("tasks")
    if tasks:
        prompts = [prompt for prompt in prompts if prompt.task.lower() in set(tasks)]
    if not prompts:
        raise ConfigError(f"dataset.tasks: нет промптов для {tasks!r}")

    n = int(dataset["n"])
    if n > len(prompts):
        raise ConfigError(
            f"dataset.n={n}, а после фильтра tasks осталось {len(prompts)} промптов"
        )

    random.Random(int(dataset.get("seed", 0))).shuffle(prompts)
    return prompts[:n]


# ---------------------------------------------------------------------------
# 4. FakeEngine с управляемой скоростью: только подкласс, исходный файл не меняем.
# ---------------------------------------------------------------------------

class SpeedControlledFakeEngine(FakeEngineV1):
    """FakeEngine, у которого задержка задаётся флагом fake_speed."""

    def __init__(self, *, speed_factor: float = 1.0, seed: int = 123,
                 name: str = "fake", flags: dict[str, Any] | None = None):
        super().__init__(seed=seed)
        self.speed_factor = float(speed_factor)
        self.name = name
        self.model = "fake-v1"
        self.flags = dict(flags or {})

    def generate(self, prompt: Prompt, gen: GenParams) -> list[Generation]:
        return self._generate(prompt, gen, self.speed_factor)

    def generate_many(self, prompts: list[Prompt], gen: GenParams,
                      parallel: int = 1) -> list[list[Generation]]:
        return [self._generate(prompt, gen, self.speed_factor) for prompt in prompts]

    def info(self) -> dict[str, Any]:
        info = super().info()
        info.update({
            "engine": "fake",
            "model": self.model,
            "variant": self.name,
            "speed_factor": self.speed_factor,
            "flags": dict(self.flags),
        })
        return info

    def __enter__(self) -> "SpeedControlledFakeEngine":
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: Any) -> None:
        self.stop()

    def stop(self) -> None:
        # Не печатаем "Engine stopped" из базового класса: сценарий должен быть тихим.
        self.running = False


def _make_engine(spec: Spec) -> SpeedControlledFakeEngine:
    flags = spec.flags
    return SpeedControlledFakeEngine(
        speed_factor=float(flags.get("fake_speed", 1.0)),
        seed=int(flags.get("fake_seed", 123)),
        name=spec.name,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# 5. Замер: прогрев, повторы, чередование вариантов.
# ---------------------------------------------------------------------------

def _rotate(items: list[Prompt], offset: int) -> list[Prompt]:
    if not items:
        return items
    offset %= len(items)
    return items[offset:] + items[:offset]


def _run_ab(engines: dict[str, SpeedControlledFakeEngine],
            specs: dict[str, Spec],
            prompts: list[Prompt],
            repeats: int,
            heartbeat: Any) -> dict[str, list[Record]]:
    records: dict[str, list[Record]] = {name: [] for name in engines}

    # Прогрев на двух промптах, в результат не идёт.
    for name, engine in engines.items():
        for prompt in prompts[:2]:
            engine.generate(prompt, specs[name].gen)

    engine_items = list(engines.items())
    for repeat in range(repeats):
        # Меняем порядок вариантов между повторами: A B, B A.
        offset = repeat % len(engine_items)
        ordered = engine_items[offset:] + engine_items[:offset]
        for name, engine in ordered:
            spec = specs[name]
            for prompt in _rotate(prompts, repeat):
                for generation in engine.generate(prompt, spec.gen):
                    records[name].append(Record(name, repeat, prompt, generation))
        heartbeat("measure", 0.2 + 0.6 * (repeat + 1) / repeats)
    return records


# ---------------------------------------------------------------------------
# 6. Статистика и сравнение: минимально достаточно для шага 1.
# ---------------------------------------------------------------------------

def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("пустой список")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - position) + ordered[high] * (position - low))


def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choice(values) for _ in range(len(values)))
        for _ in range(n_boot)
    ]
    return [
        round(_percentile(medians, 0.025), 6),
        round(_percentile(medians, 0.975), 6),
    ]


def _speed_stats(records: list[Record]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("нет генераций для статистики")
    speeds = [record.generation.tok_s for record in records if record.generation.tok_s > 0]
    if not speeds:
        raise RuntimeError("нет ни одной генерации с tok_s > 0")

    ttft = [record.generation.ttft_ms for record in records]
    total_ms = [record.generation.total_ms for record in records]
    draft_n = sum(record.generation.draft_n for record in records)
    accepted = sum(record.generation.draft_accepted for record in records)
    n_gen = sum(record.generation.n_gen for record in records)
    passes = sum(record.generation.n_gen - record.generation.draft_accepted for record in records)

    return {
        "tok_s_median": round(statistics.median(speeds), 6),
        "tok_s_p10": round(_percentile(speeds, 0.10), 6),
        "tok_s_p90": round(_percentile(speeds, 0.90), 6),
        "ttft_ms_median": round(statistics.median(ttft), 6),
        "total_s_median": round(statistics.median(total_ms) / 1000.0, 6),
        "acceptance": round(accepted / draft_n, 6) if draft_n else None,
        "tokens_per_pass": round(n_gen / passes, 6) if accepted > 0 and passes > 0 else None,
        "frac_of_ceiling": None,
    }


def _first_generations(records: list[Record]) -> dict[str, Generation]:
    result: dict[str, Generation] = {}
    for record in sorted(records, key=lambda item: item.repeat):
        result.setdefault(record.prompt.id, record.generation)
    return result


def _median_by_prompt(records: list[Record], attr: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.prompt.id].append(float(getattr(record.generation, attr)))
    return {prompt_id: statistics.median(values) for prompt_id, values in grouped.items()}


def _top2_margin(generation: Generation, position: int) -> float:
    if not generation.logprobs or position < 0 or position >= len(generation.logprobs):
        return 0.0
    values = sorted((float(logprob) for _, logprob in generation.logprobs[position]), reverse=True)
    if len(values) < 2:
        return 0.0
    return max(0.0, values[0] - values[1])


def _tokens_equal(left: Generation, right: Generation) -> bool:
    if left.token_ids or right.token_ids:
        return left.token_ids == right.token_ids
    return left.text == right.text


def _compare(base: Spec, variant: Spec,
             base_records: list[Record], variant_records: list[Record]) -> dict[str, Any]:
    base_speed = _median_by_prompt(base_records, "tok_s")
    variant_speed = _median_by_prompt(variant_records, "tok_s")
    common = sorted(set(base_speed) & set(variant_speed))

    ratios = [
        variant_speed[prompt_id] / base_speed[prompt_id]
        for prompt_id in common
        if base_speed[prompt_id] > 0
    ]
    speedup = round(statistics.median(ratios), 6) if ratios else None
    speedup_ci = _bootstrap_ci(ratios) if ratios else None

    base_total = _median_by_prompt(base_records, "total_ms")
    variant_total = _median_by_prompt(variant_records, "total_ms")
    total_ratios = [
        base_total[prompt_id] / variant_total[prompt_id]
        for prompt_id in common
        if variant_total[prompt_id] > 0
    ]
    total_time_ratio = round(statistics.median(total_ratios), 6) if total_ratios else None

    base_first = _first_generations(base_records)
    variant_first = _first_generations(variant_records)
    divergences: list[dict[str, Any]] = []
    for prompt_id in common:
        base_gen = base_first[prompt_id]
        variant_gen = variant_first[prompt_id]
        if _tokens_equal(base_gen, variant_gen):
            continue
        limit = min(len(base_gen.token_ids), len(variant_gen.token_ids))
        position = next(
            (index for index in range(limit)
             if base_gen.token_ids[index] != variant_gen.token_ids[index]),
            limit,
        )
        divergences.append({
            "prompt_id": prompt_id,
            "position": position,
            "base_margin": round(_top2_margin(base_gen, position), 6),
        })

    exact_match = (
        round((len(common) - len(divergences)) / len(common), 6)
        if common else None
    )
    numeric_divergences = sum(1 for item in divergences if item["base_margin"] < 0.1)

    base_tokens = [record.generation.n_gen for record in base_records]
    variant_tokens = [record.generation.n_gen for record in variant_records]
    tokens_ratio = (
        round(statistics.fmean(variant_tokens) / statistics.fmean(base_tokens), 6)
        if base_tokens and variant_tokens and statistics.fmean(base_tokens) else None
    )

    if speedup is None:
        verdict = "нет данных"
    elif exact_match is not None and exact_match < 0.99:
        verdict = "быстрее, но ответы расходятся" if speedup >= 1.05 else "медленнее"
    elif speedup >= 1.05:
        verdict = "быстрее, без потерь"
    elif speedup <= 0.95:
        verdict = "медленнее"
    else:
        verdict = "не значимо"

    warnings: list[str] = []
    if exact_match is not None and exact_match < 1.0:
        warnings.append(
            "ответы расходятся с базой; смотри divergences и base_margin "
            "(маленький отрыв — похоже на округление)"
        )

    return {
        "variant": variant.name,
        "base": base.name,
        "speedup": speedup,
        "speedup_ci95": speedup_ci,
        "total_time_ratio": total_time_ratio,
        "exact_match": exact_match,
        "divergences": divergences,
        "numeric_divergences": numeric_divergences,
        "accuracy_delta": None,
        "accuracy_delta_ci95": None,
        "tokens_ratio": tokens_ratio,
        "cost_ratio": None,
        "verdict": verdict,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 7. Устройство и запись папки результата.
# ---------------------------------------------------------------------------

def _ram_gb() -> float:
    try:
        if sys.platform == "darwin":
            value = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(value.strip()) / (1024 ** 3), 3)
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 3)
    except Exception:
        pass
    return 0.0


def _fastinfer_version() -> str:
    try:
        from importlib.metadata import version
        return version("fastinfer")
    except Exception:
        return "unknown"


def _device_info() -> DeviceInfo:
    return DeviceInfo(
        kind=os.environ.get("FASTINFER_DEVICE", "pc-cpu"),
        backend="cpu",
        gpus=[],
        cpu=platform.processor() or platform.machine(),
        cores=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
        os=platform.platform(),
        bandwidth_gbs=0.0,
        versions={
            "python": platform.python_version(),
            "fastinfer": _fastinfer_version(),
            "platform": platform.platform(),
        },
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _atomic_write_text(path, text)


def _write_heartbeat(path: Path, stage: str, progress: float) -> None:
    _atomic_write_json(path, {
        "time": _now(),
        "stage": stage,
        "progress": round(max(0.0, min(1.0, float(progress))), 3),
        "alive": True,
    })


def _log(out_dir: Path, message: str) -> None:
    path = out_dir / "logs" / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{_now()} {message}\n")


def _margins(generation: Generation) -> list[float] | None:
    if not generation.logprobs:
        return None
    result: list[float] = []
    for position in generation.logprobs:
        values = sorted((float(logprob) for _, logprob in position), reverse=True)
        result.append(round(values[0] - values[1], 6) if len(values) >= 2 else 0.0)
    return result


def _config_hash(resolved: dict[str, Any]) -> str:
    canonical = json.dumps(
        resolved,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_outputs(*, out_dir: Path, resolved: dict[str, Any], base: Spec,
                   variants: list[Spec], prompts: list[Prompt],
                   records: dict[str, list[Record]], engines_info: dict[str, dict[str, Any]],
                   device: DeviceInfo, job_id: str, commit: str,
                   started_at: str, finished_at: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = out_dir / "heartbeat.json"
    _write_heartbeat(heartbeat_path, "write", 0.85)

    per_prompt_path = out_dir / "per_prompt.jsonl"
    lines: list[str] = []
    for spec in [base, *variants]:
        for record in records[spec.name]:
            generation = record.generation
            row: dict[str, Any] = {
                "variant": spec.name,
                "repeat": record.repeat,
                "prompt_id": generation.prompt_id,
                "task": record.prompt.task,
                "turn": generation.turn,
                "n_prompt": generation.n_prompt,
                "n_gen": generation.n_gen,
                "n_thinking": generation.n_thinking,
                "n_answer": generation.n_answer,
                "ttft_ms": generation.ttft_ms,
                "decode_ms": generation.decode_ms,
                "total_ms": generation.total_ms,
                "tok_s": generation.tok_s,
                "draft_n": generation.draft_n,
                "draft_accepted": generation.draft_accepted,
                "finish_reason": generation.finish_reason,
                "correct": None,
                "cost_usd": generation.cost_usd,
                "text": generation.text,
            }
            if record.repeat == 0:
                row["token_ids"] = generation.token_ids
                row["top2_margin"] = _margins(generation)
            lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    _atomic_write_text(per_prompt_path, "\n".join(lines) + ("\n" if lines else ""))

    versions = {
        "python": platform.python_version(),
        "fastinfer": _fastinfer_version(),
        "platform": platform.platform(),
    }
    environment = {
        key: os.environ[key]
        for key in (
            "LLAMA_CPP_BIN",
            "FASTINFER_CACHE",
            "FASTINFER_JOB_ID",
            "FASTINFER_COMMIT",
            "FASTINFER_DEVICE",
            "FASTINFER_API_CAP_USD",
        )
        if key in os.environ
    }
    env_payload = {
        "schema": 1,
        "job_id": job_id,
        "commit": commit,
        "name": resolved["name"],
        "family": resolved["family"],
        "mode": resolved["mode"],
        "started_at": started_at,
        "config": resolved,
        "config_hash": _config_hash(resolved),
        "engine": {"variants": engines_info},
        "device": asdict(device),
        "models": {},
        "caps": {
            "budget_usd": resolved["budget_usd"],
            "timeout_min": resolved["timeout_min"],
            "ceiling_tok_s": None,
        },
        "environment": environment,
        "versions": versions,
    }
    _atomic_write_json(out_dir / "env.json", env_payload)

    variants_payload: dict[str, Any] = {}
    for spec in [base, *variants]:
        variants_payload[spec.name] = {
            "variant": spec.name,
            "mode": resolved["mode"],
            "speed": _speed_stats(records[spec.name]),
            "quality": None,
            "engine_info": engines_info[spec.name],
        }

    comparison = {
        variant.name: _compare(base, variant, records[base.name], records[variant.name])
        for variant in variants
    }

    warnings = [
        "FakeEngine: ответы и тайминги синтетические; это проверка среза, а не замер.",
        "dataset: используются встроенные 4 промпта, сеть не нужна.",
        "device: описано условно как pc-cpu; реальные GPU/ПСП появятся на шаге 6.",
    ]

    result = {
        "schema": 1,
        "job_id": job_id,
        "commit": commit,
        "name": resolved["name"],
        "family": resolved["family"],
        "mode": resolved["mode"],
        "device": asdict(device),
        "started_at": started_at,
        "finished_at": finished_at,
        "ceiling_tok_s": None,
        "variants": variants_payload,
        "comparison": comparison,
        "accept": {},
        "analysis": {},
        "cost_usd": 0.0,
        "warnings": warnings,
    }
    _atomic_write_json(out_dir / "result.json", result)
    _write_heartbeat(heartbeat_path, "done", 1.0)
    return result


# ---------------------------------------------------------------------------
# 8. Сквозной запуск и CLI.
# ---------------------------------------------------------------------------

def run(config_path: str | Path | None = DEFAULT_CONFIG_PATH,
        out_dir: str | Path = "run_results/fake_smoke",
        job_id: str | None = None, commit: str | None = None,
        verbose: bool = False) -> RunArtifacts:
    started_at = _now()
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    heartbeat_path = out / "heartbeat.json"
    _write_heartbeat(heartbeat_path, "start", 0.0)
    _log(out, "start")
    started = time.perf_counter()

    resolved, base, variants, prompts = load_and_resolve_config(config_path)
    _write_heartbeat(heartbeat_path, "config", 0.1)
    _log(out, f"config resolved: {len(prompts)} prompts, base={base.name}, "
               f"variants={[variant.name for variant in variants]}")

    specs = [base, *variants]
    specs_by_name = {spec.name: spec for spec in specs}
    records: dict[str, list[Record]] = {}
    engines_info: dict[str, dict[str, Any]] = {}

    with ExitStack() as stack:
        engines: dict[str, SpeedControlledFakeEngine] = {}
        for spec in specs:
            engine = stack.enter_context(_make_engine(spec))
            engines[spec.name] = engine
            engines_info[spec.name] = engine.info()
            _log(out, f"engine started: {spec.name} {engine.info()}")

        _write_heartbeat(heartbeat_path, "engines", 0.2)
        records = _run_ab(
            engines,
            specs_by_name,
            prompts,
            repeats=int(resolved["repeats"]),
            heartbeat=lambda stage, progress: _write_heartbeat(heartbeat_path, stage, progress),
        )

    _write_heartbeat(heartbeat_path, "write", 0.85)
    finished_at = _now()
    result = _write_outputs(
        out_dir=out,
        resolved=resolved,
        base=base,
        variants=variants,
        prompts=prompts,
        records=records,
        engines_info=engines_info,
        device=_device_info(),
        job_id=job_id or os.environ.get("FASTINFER_JOB_ID") or f"fake-{int(time.time())}",
        commit=commit or os.environ.get("FASTINFER_COMMIT") or "local",
        started_at=started_at,
        finished_at=finished_at,
    )

    elapsed_s = time.perf_counter() - started
    _log(out, f"done in {elapsed_s:.3f}s")
    if verbose:
        print(f"[fastinfer-fake] result: {out / 'result.json'}")
    return RunArtifacts(
        result=result,
        out_dir=out,
        base_name=base.name,
        n_prompts=len(prompts),
        n_variants=len(specs),
        repeats=int(resolved["repeats"]),
        elapsed_s=elapsed_s,
    )


def _summary_line(artifacts: RunArtifacts) -> str:
    result = artifacts.result
    base_speed = result["variants"][artifacts.base_name]["speed"]["tok_s_median"]
    comparisons = result["comparison"]
    if not comparisons:
        return f"{result['name']}: base {base_speed:.1f} tok/s, сравнений нет"
    variant_name, comparison = next(iter(comparisons.items()))
    speedup = comparison["speedup"]
    exact = comparison["exact_match"]
    ci = comparison["speedup_ci95"]
    speedup_text = "n/a" if speedup is None else f"{speedup:.2f}"
    exact_text = "n/a" if exact is None else f"{exact:.2f}"
    ci_text = "" if not ci else f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return (
        f"{result['name']}: base {base_speed:.1f} tok/s; "
        f"{variant_name} x{speedup_text}{ci_text}; exact={exact_text}; "
        f"verdict={comparison['verdict']!r}"
    )


def _self_check(artifacts: RunArtifacts, *, strict: bool = True) -> None:
    required = ("result.json", "per_prompt.jsonl", "env.json", "heartbeat.json")
    missing = [name for name in required if not (artifacts.out_dir / name).is_file()]
    if missing:
        raise SelfCheckError(f"нет файлов результата: {missing}")

    for name in ("result.json", "env.json", "heartbeat.json"):
        try:
            json.loads((artifacts.out_dir / name).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SelfCheckError(f"{name}: невалидный JSON: {exc}") from exc

    result = json.loads((artifacts.out_dir / "result.json").read_text(encoding="utf-8"))
    if len(result.get("variants", {})) < 2:
        raise SelfCheckError("в результате меньше двух вариантов")
    if not result.get("comparison"):
        raise SelfCheckError("в результате нет сравнения с базой")

    if strict:
        if artifacts.n_variants != 2:
            raise SelfCheckError(f"вариантов {artifacts.n_variants}, ожидалось 2 (база + вариант)")
        if artifacts.n_prompts != 4:
            raise SelfCheckError(f"промптов {artifacts.n_prompts}, ожидалось 4")
        if artifacts.repeats != 2:
            raise SelfCheckError(f"повторов {artifacts.repeats}, ожидалось 2")
        if artifacts.elapsed_s >= 5.0:
            raise SelfCheckError(f"сценарий шёл {artifacts.elapsed_s:.2f}s, ожидалось < 5s")
        lines = (artifacts.out_dir / "per_prompt.jsonl").read_text(encoding="utf-8").splitlines()
        expected = artifacts.n_variants * artifacts.n_prompts * artifacts.repeats
        if len(lines) != expected:
            raise SelfCheckError(f"строк per_prompt {len(lines)}, ожидалось {expected}")

    print(
        f"SELF-CHECK OK: {artifacts.n_variants} варианта, "
        f"{artifacts.n_prompts} промпта, {artifacts.repeats} повтора, "
        f"{artifacts.elapsed_s:.3f}s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Сквозной сценарий шага 1 на FakeEngine",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML/JSON конфига; по умолчанию core/resources/test_scenario_fake.yaml",
    )
    parser.add_argument("--out", default="run_results/fake_smoke", help="папка результата")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    strict = Path(args.config).expanduser().resolve() == DEFAULT_CONFIG_PATH.resolve()

    try:
        artifacts = run(
            config_path=args.config,
            out_dir=args.out,
            job_id=args.job_id,
            commit=args.commit,
            verbose=args.verbose,
        )
        _self_check(artifacts, strict=strict)
        print(_summary_line(artifacts))
        print(f"OUT: {artifacts.out_dir.resolve()}")
        return 0
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except SelfCheckError as exc:
        print(f"SELF-CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # движок/запись — отдельный код, чтобы было видно в запуске
        print(f"RUN FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 5


def test_fake_scenario() -> None:
    """Сценарий без аргументов: вызывается из main_tests или pytest.

    Конфиг берётся из core/resources/test_scenario_fake.yaml. Результат пишется
    во временную папку и удаляется после проверки: 2 варианта, 4 промпта,
    2 повтора, без сети.
    """
    from fastinfer.config_load import load_yaml, validate_config

    config_errors = validate_config(load_yaml(DEFAULT_CONFIG_PATH))
    assert not config_errors, f"config errors: {config_errors}"

    with tempfile.TemporaryDirectory(prefix="fastinfer-fake-") as tmp:
        artifacts = run(
            config_path=DEFAULT_CONFIG_PATH,
            out_dir=Path(tmp) / "result",
            job_id="test-fake-scenario",
            commit="local",
        )
        _self_check(artifacts, strict=True)
        assert artifacts.n_variants == 2
        assert artifacts.n_prompts == 4
        assert artifacts.repeats == 2
        assert artifacts.elapsed_s < 5.0
        assert artifacts.result["comparison"]["faster"]["speedup"] > 1.0


if __name__ == "__main__":
    raise SystemExit(main())

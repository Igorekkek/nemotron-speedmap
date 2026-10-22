"""Сквозной сценарий этапа 0 на FakeEngine.

Путь ровно как в шаге 1:
    конфиг -> база + вариант -> FakeEngine -> генерации -> скорость -> папка результата.

Запускается как функция из core/tests/main_tests.py:
    from tests.test_scenario_fake import test_fake_scenario
    test_fake_scenario()

Конфиг лежит в core/resources/test_scenario_fake.yaml, промпты — в
core/resources/local.jsonl. Сеть не нужна, реальные модели не загружаются.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence

from core.fastinfer import ExperimentConfig, GenParams, Prompt, Report, VariantSpec
from core.fastinfer.config_load import load_experiment_config
from core.fastinfer.engines.fake import FakeEngineV1
from core.fastinfer.measure import measure_speed
from core.fastinfer.raw_data import load_prompts
from core.fastinfer.results import ExperimentWriter

RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"
CONFIG_PATH = RESOURCES_DIR / "test_scenario_fake.yaml"


def _gen_for(config: ExperimentConfig, spec: VariantSpec) -> GenParams:
    """Параметры генерации варианта: поля spec.gen поверх gen из конфига."""
    if not spec.gen:
        return config.gen
    return dataclasses.replace(config.gen, **spec.gen)


def _fake_engine(spec: VariantSpec, base: VariantSpec) -> FakeEngineV1:
    """Собрать FakeEngine по флагам варианта; незаданные флаги наследуются от базы."""
    flags = {**base.flags, **spec.flags}
    engine = FakeEngineV1(
        speed=float(flags.get("fake_speed", 1.0)),
        seed=int(flags.get("fake_seed", 123)),
    )
    engine.name = spec.name
    return engine


def _measure_spec(config: ExperimentConfig, spec: VariantSpec,
                  base: VariantSpec, prompts: Sequence[Prompt]) -> Report:
    """Замерить один вариант уже реализованным measure_speed."""
    with _fake_engine(spec, base) as engine:
        return measure_speed(
            engine,
            prompts,
            _gen_for(config, spec),
            repeats=config.repeats,
        )


def _write_report(writer: ExperimentWriter, report: Report,
                  prompts: Sequence[Prompt]) -> None:
    """Записать генерации варианта в per_prompt.jsonl."""
    tasks = {prompt.id: prompt.task for prompt in prompts}
    for index, generation in enumerate(report.generations):
        writer.write_perprompt(
            variant=report.variant,
            repeat=index // len(prompts),
            generation=generation,
            task=tasks[generation.prompt_id],
        )


def test_fake_scenario(config_path: str | Path = CONFIG_PATH) -> None:
    """Проверить сквозной путь на FakeEngine и записать папку результата."""
    config = load_experiment_config(str(config_path))

    prompts = load_prompts(
        config.dataset.name,
        n=config.dataset.n,
        seed=config.dataset.seed,
        tasks=config.dataset.tasks,
    )
    assert prompts, "датасет пуст"
    assert config.dataset.n is None or len(prompts) == config.dataset.n

    writer = ExperimentWriter(config)
    writer.start()
    writer.write_log(
        f"scenario: {config.name}, prompts={len(prompts)}, repeats={config.repeats}\n"
    )

    reports: dict[str, Report] = {
        config.base.name: _measure_spec(config, config.base, config.base, prompts),
    }
    for variant in config.variants:
        reports[variant.name] = _measure_spec(config, variant, config.base, prompts)

    expected_generations = config.repeats * len(prompts)
    for report in reports.values():
        assert report.speed is not None
        assert len(report.generations) == expected_generations
        assert all(
            generation.n_gen == len(generation.token_ids) > 0
            for generation in report.generations
        )
        assert report.speed.tok_s_median > 0
        _write_report(writer, report, prompts)

    base_report = reports[config.base.name]
    for name, report in reports.items():
        if name == config.base.name:
            continue
        assert report.speed.tok_s_median > base_report.speed.tok_s_median, (
            f"{name}: ожидали ускорение относительно {config.base.name}"
        )

    writer.write_env(engine_info=base_report.engine_info)
    writer.write_log("fake scenario passed\n")

    assert writer.pth_env_file.is_file()
    assert writer.pth_perprompt_file.is_file()

    speeds = ", ".join(
        f"{name}={report.speed.tok_s_median:.1f} tok/s"
        for name, report in reports.items()
    )
    print(
        f"fake scenario passed: {config.name}, "
        f"prompts={len(prompts)}, repeats={config.repeats}, {speeds}"
    )

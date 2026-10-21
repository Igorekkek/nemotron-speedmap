"""fastinfer — общее ядро экспериментов по ускорению инференса Nemotron.

Здесь функции для исследователя — docs/core_api.md, раздел 7.1. Пока это только контракт: сигнатуры
и описания, тела — `raise NotImplementedError`. Реализацию пишет разработчик ядра; остальные модули
(config.py, measure.py, engines/llamacpp.py и т. д., раздел 2) он заводит сам.
Сигнатуры — общий контракт команды: менять — только через PR, который смотрят все трое.
"""

from pathlib import Path

from .engines import Engine
from .types import (
    AcceptStats,
    Comparison,
    DatasetSpec,
    DeviceInfo,
    Divergence,
    ExperimentConfig,
    Generation,
    GenParams,
    ModelSpec,
    Prompt,
    QualityStats,
    Report,
    RunResult,
    SpeedStats,
    VariantSpec,
)

__all__ = [
    # функции для исследователя (раздел 7.1)
    "load_config",
    "run_experiment",
    "load_model",
    "load_prompts",
    "generate",
    "measure",
    "compare",
    "save_results",
    "device_info",
    "ceiling",
    # интерфейс движка (раздел 6)
    "Engine",
    # типы (раздел 5)
    "Prompt",
    "GenParams",
    "Generation",
    "SpeedStats",
    "QualityStats",
    "Report",
    "Divergence",
    "Comparison",
    "AcceptStats",
    "DeviceInfo",
    "ModelSpec",
    "DatasetSpec",
    "VariantSpec",
    "ExperimentConfig",
    "RunResult",
]


def load_config(path: str | Path) -> ExperimentConfig:
    """Прочитать и проверить config.yaml эксперимента.

    Принимает:
        path: путь к config.yaml (формат — раздел 3).

    Возвращает:
        ExperimentConfig.

    Ошибки в конфиге ищутся все сразу и до загрузки моделей (validate_config, раздел 7.2): неизвестный
    флаг движка, недопустимые mode, device или engine, нет обязательного поля. Незаданные поля варианта
    берутся из базы, а у базы — из верхнего уровня (раздел 3); перебор (sweep) разворачивает
    expand_variants (раздел 7.2).

    Этап 0. См. docs/core_api.md, разделы 3 и 7.1.
    """
    raise NotImplementedError


def run_experiment(cfg: ExperimentConfig, out_dir: str | Path) -> RunResult:
    """Выполнить конфиг целиком и записать папку результата.

    Принимает:
        cfg: конфиг из load_config.
        out_dir: куда писать папку результата (раздел 4).

    Возвращает:
        RunResult — то же, что попадёт в result.json.

    Это то, что вызывает `python -m fastinfer.run` (раздел 11). Внутри: варианты с перебором, промпты,
    прогрев, замер базы и вариантов по правилам раздела 9 (один запуск, одно устройство, повторы
    чередуются), сравнение каждого варианта с базой, запись результата и heartbeat.json. Соблюдает
    timeout_min (пишет то, что успело) и потолок денег API (budget_usd и FASTINFER_API_CAP_USD).

    Этап 0. См. docs/core_api.md, разделы 4, 7.1, 9 и 11.
    """
    raise NotImplementedError


def load_model(name: str, engine: str = "llamacpp", quant: str | None = None,
               flags: dict | None = None) -> Engine:
    """Подготовить движок с моделью.

    Принимает:
        name: имя модели из реестра, например "nano-9b-v2" (раздел 3, «Начальный реестр моделей»).
        engine: "llamacpp", "hf" или "api".
        quant: для llamacpp — квант GGUF ("Q4_K_M"…), для hf — dtype ("fp16", "bf16", "int8", "nf4").
        flags: флаги движка — ключи без дефисов, например {"ngl": 99, "spec_type": "ngram-mod"}
            (раздел 3, таблица флагов llama.cpp). На неизвестном ключе — ошибка до загрузки модели.

    Возвращает:
        Engine, который работает как контекстный менеджер: в `with load_model(...) as m:`
        `start` и `stop` вызываются сами.

    Этап 0. См. docs/core_api.md, разделы 3, 6, 7.1 и 8.
    """
    raise NotImplementedError


def load_prompts(dataset: str, n: int | None = None, seed: int = 0,
                 tasks: list[str] | None = None) -> list[Prompt]:
    """Загрузить набор промптов.

    Принимает:
        dataset: имя набора — "spec_bench", "gsm8k" (этап 0), "math500", "aime" (этап 1).
        n: сколько промптов взять; None — все.
        seed: зерно выборки, чтобы при одном seed выборка была одной и той же.
        tasks: взять только эти задачи (например, категории Spec-Bench); None — все.

    Возвращает:
        Список Prompt.

    Этап 0. См. docs/core_api.md, разделы 7.1 и 7.4.
    """
    raise NotImplementedError


def generate(engine: Engine, prompts: list[Prompt], gen: GenParams,
             parallel: int = 1) -> list[list[Generation]]:
    """Сгенерировать ответы без правил замера — для своего кода исследователя.

    Принимает:
        engine: движок из load_model.
        prompts: промпты.
        gen: параметры генерации.
        parallel: сколько промптов генерировать одновременно.

    Возвращает:
        По списку Generation на промпт — по одной генерации на ход.

    Этап 0. См. docs/core_api.md, разделы 6 и 7.1.
    """
    raise NotImplementedError


def measure(engine: Engine, prompts: list[Prompt], gen: GenParams, mode: str = "speed",
            repeats: int = 3) -> Report:
    """Замерить один движок по правилам раздела 9.

    Принимает:
        engine: движок из load_model.
        prompts: промпты.
        gen: параметры генерации.
        mode: "speed" — батч 1, жадный выбор, прогрев не считается, повторы (этап 0);
            "quality" — генерация батчем и проверка ответов (этап 1).
        repeats: число повторов для "speed"; в "quality" всегда 1.

    Возвращает:
        Report: в "speed" заполнено speed, в "quality" — quality.

    measure мерит один движок, поэтому повторы разных вариантов здесь не чередуются (раздел 9, п. 3).
    Для честного A/B нескольких вариантов есть run_experiment.

    Этап 0 (speed), этап 1 (quality). См. docs/core_api.md, разделы 7.1, 7.6 и 9.
    """
    raise NotImplementedError


def compare(base: Report, variant: Report) -> Comparison:
    """Сравнить вариант с базой.

    Принимает:
        base: Report базы.
        variant: Report варианта на тех же промптах.

    Возвращает:
        Comparison: ускорение — медиана отношений tok_s по одним и тем же промптам, интервал — бутстрепом;
        совпадение ответов с базой и места расхождений с отрывом top-2 у базы (раздел 9, п. 6);
        для качества — разница точности; вердикт.

    Этап 0. См. docs/core_api.md, разделы 5, 7.1 и 9.
    """
    raise NotImplementedError


def save_results(result: RunResult, out_dir: str | Path) -> dict[str, Path]:
    """Записать папку результата.

    Принимает:
        result: результат запуска.
        out_dir: папка результата.

    Возвращает:
        Записанные файлы: имя → путь, например {"result.json": Path(".../result.json"), ...}.

    Формат папки — раздел 4: в result.json варианты без generations, генерации — в per_prompt.jsonl.

    Этап 0. См. docs/core_api.md, разделы 4 и 7.1.
    """
    raise NotImplementedError


def device_info() -> DeviceInfo:
    """Описать текущее устройство.

    Возвращает:
        DeviceInfo: тип устройства (или FASTINFER_DEVICE, если его не определить надёжно), бэкенд,
        видеокарты, процессор, память, ОС, пиковая ПСП, версии библиотек.

    Этап 0. См. docs/core_api.md, разделы 5, 7.1 и 10.
    """
    raise NotImplementedError


def ceiling(model: str, quant: str, device: DeviceInfo | str | None = None) -> float:
    """Потолок скорости генерации, ток/с: пиковая ПСП ÷ байты, прочитанные на токен.

    Принимает:
        model: имя модели из реестра.
        quant: квант, например "Q4_K_M".
        device: DeviceInfo, тип устройства ("kaggle-t4x2"…) или None — текущее устройство.

    Возвращает:
        Потолок, ток/с. Для проверки: nano-9b-v2 в Q4_K_M на T4 — около 64 (docs/families.md).

    Этап 0. См. docs/core_api.md, разделы 7.1 и 7.3; docs/families.md, «Откуда берётся время».
    """
    raise NotImplementedError



































































































































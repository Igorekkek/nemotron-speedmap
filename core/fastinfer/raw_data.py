"""Загрузка датасетов и подготовка сообщений — docs/core_api.md, разделы 7.4 и 13.

Возможные источники датасетов:

- ``spec_bench`` — JSONL из GitHub ``hemingkx/Spec-Bench`` (data/spec_bench/question.jsonl),
  закреплённая ревизия + sha256; 480 вопросов, 80 многоходовых.
- ``gsm8k`` — HF ``openai/gsm8k`` через datasets-server (config=main, split=test),
  без зависимости от ``datasets``/``pyarrow``; кэшируется в JSONL.
- ``local:<name>`` / ``<name>.jsonl`` — локальный JSONL из ``core/resources``.
  Поддерживаются записи двух видов:
    * Prompt-подобные: ``id``, ``task``/``category``, ``turns``, ``reference``, ``meta``;
    * GSM8K-подобные: ``question``, ``answer``, необязательные ``id``/``idx``.
- ``math500``, ``aime`` — этап 1, в реестре стоят заглушки.

Сетевые датасеты кэшируются в ``FASTINFER_CACHE/datasets/<name>/<split>.jsonl``
(по умолчанию ``~/.cache/fastinfer``, на Kaggle — ``/kaggle/temp/fastinfer``).
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from fastinfer import GenParams, ModelSpec, Prompt

RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"
LOCAL_PREFIX = "local:"
GSM8K_ROWS_URL = "https://datasets-server.huggingface.co/rows"

Message = dict[str, str]


@dataclass(frozen=True)
class DatasetSource:
    name: str
    stage: int                 # 0 — этап 0, 1 — этап 1
    kind: str                  # "jsonl_url" | "hf_rows" | "local_jsonl" | "todo"
    location: str              # URL, repo_id или имя файла в resources
    revision: str | None = None
    sha256: str | None = None
    config: str | None = None
    split: str | None = None


DATASETS: dict[str, DatasetSource] = {
    "spec_bench": DatasetSource(
        name="spec_bench",
        stage=0,
        kind="jsonl_url",
        location=(
            "https://raw.githubusercontent.com/hemingkx/Spec-Bench/"
            "{revision}/data/spec_bench/question.jsonl"
        ),
        revision="main",       # заменить на commit
        sha256=None,           # вписать после первого скачивания
    ),
    "gsm8k": DatasetSource(
        name="gsm8k",
        stage=0,
        kind="hf_rows",
        location="openai/gsm8k",
        config="main",
        split="test",
        sha256=None,
    ),
    "math500": DatasetSource(name="math500", stage=1, kind="todo", location=""),
    "aime": DatasetSource(name="aime", stage=1, kind="todo", location=""),
}


class DataException(Exception):

    def __init__(self, err: str):
        super().__init__(err)


def load_prompts(dataset: str, n: int | None = None, seed: int = 0,
                 tasks: list[str] | None = None) -> list[Prompt]:
    """Загрузить набор промптов и детерминированно выбрать из него `n` штук.

    `dataset` — имя из DATASETS или локальный файл: ``local:<name>`` / ``<name>.jsonl``
    (файл ищется в ``core/resources``). Один и тот же (dataset, n, seed, tasks) даёт
    одинаковый список на всех машинах.
    """
    source = DATASETS.get(dataset)

    if source is None:
        source = _local_source(dataset)
        if source is None:
            raise DataException(
                f"unknown dataset: {dataset}; known: {', '.join(sorted(DATASETS))}, "
                f"или локальный файл в {RESOURCES_DIR}"
            )

    if source.stage > 0:
        raise DataException(f"{dataset}: planned for stage {source.stage}, not stage 0")

    if source.kind == "local_jsonl":
        prompts = _load_local(source)
    elif source.kind == "jsonl_url":
        prompts = _load_spec_bench(source)
    elif source.kind == "hf_rows":
        prompts = _load_gsm8k(source)
    else:
        raise DataException(f"{dataset}: no loader for kind {source.kind!r}")

    return _select(prompts, n=n, seed=seed, tasks=tasks)


def _local_source(name: str) -> DatasetSource | None:
    """Найти локальный JSONL в core/resources по имени или local:<имя>."""
    raw = name[len(LOCAL_PREFIX):] if name.startswith(LOCAL_PREFIX) else name
    raw = raw.strip()
    if not raw:
        return None

    filename = raw if raw.endswith(".jsonl") else f"{raw}.jsonl"
    if Path(filename).name != filename:
        raise DataException(
            f"local dataset must be a file name inside {RESOURCES_DIR}: {name!r}"
        )

    path = RESOURCES_DIR / filename
    if not path.is_file():
        return None

    return DatasetSource(
        name=path.stem,
        stage=0,
        kind="local_jsonl",
        location=filename,
    )


def _load_local(source: DatasetSource) -> list[Prompt]:
    """Прочитать локальный JSONL из core/resources."""
    path = RESOURCES_DIR / source.location
    prompts: list[Prompt] = []
    seen: set[str] = set()

    for index, row in enumerate(_read_jsonl(path)):
        where = f"{path}:{index + 1}"

        if "turns" in row:
            turns = _normalize_turns(row["turns"], where)
            task = str(row.get("task") or row.get("category") or "local").strip().lower()
            raw_id = row.get("id") or row.get("question_id") or index
            prompt_id = _local_prompt_id(source, raw_id)
            reference = row.get("reference")
            if reference is not None:
                reference = str(reference)
        elif "question" in row:
            question = str(row["question"]).strip()
            if not question:
                raise DataException(f"{where}: empty question")
            turns = [question]
            task = str(row.get("task") or "local").strip().lower()
            raw_id = row.get("id") or row.get("idx") or index
            prompt_id = _local_prompt_id(source, raw_id)
            reference = row.get("reference")
            if reference is None:
                answer = row.get("answer")
                reference = _gsm8k_final_answer(str(answer)) if answer is not None else None
            elif reference is not None:
                reference = str(reference)
        else:
            raise DataException(
                f"{where}: expected either 'turns' or 'question' field"
            )

        if prompt_id in seen:
            raise DataException(f"{where}: duplicate prompt id {prompt_id!r}")
        seen.add(prompt_id)

        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        prompts.append(
            Prompt(
                id=prompt_id,
                task=task,
                turns=turns,
                reference=reference,
                meta={**meta, "source": "local", "file": source.location, "index": index},
            )
        )

    return prompts


def _local_prompt_id(source: DatasetSource, raw_id: object) -> str:
    value = str(raw_id)
    if value.startswith("local/"):
        return value
    return f"local/{source.name}/{value}"


def _load_spec_bench(source: DatasetSource) -> list[Prompt]:
    """Прочитать question.jsonl Spec-Bench и превратить записи в Prompt."""
    path = _ensure_jsonl(source, _download_spec_bench)
    prompts: list[Prompt] = []
    seen: set[str] = set()

    for index, row in enumerate(_read_jsonl(path)):
        where = f"{path}:{index + 1}"
        question_id = str(_required(row, "question_id", where))
        category = str(row.get("category") or "unknown").strip().lower()
        turns = _normalize_turns(row.get("turns"), where)

        prompt_id = f"spec_bench/{category}/{question_id}"
        if prompt_id in seen:
            raise DataException(f"{where}: duplicate prompt id {prompt_id!r}")
        seen.add(prompt_id)

        prompts.append(
            Prompt(
                id=prompt_id,
                task=category,
                turns=turns,
                reference=None,          # эталонных ответов в Spec-Bench нет
                meta={
                    "source": "spec_bench",
                    "index": index,
                    "category": category,
                },
            )
        )

    return prompts


def _load_gsm8k(source: DatasetSource) -> list[Prompt]:
    """Прочитать кэш GSM8K (JSONL с question/answer) и превратить записи в Prompt."""
    path = _ensure_jsonl(source, _download_gsm8k)
    split = source.split or "test"
    prompts: list[Prompt] = []
    seen: set[str] = set()

    for position, row in enumerate(_read_jsonl(path)):
        where = f"{path}:{position + 1}"
        question = str(_required(row, "question", where)).strip()
        answer = str(_required(row, "answer", where))
        if not question:
            raise DataException(f"{where}: empty question")
        row_index = row.get("idx", position)

        prompt_id = f"gsm8k/{split}/{row_index}"
        if prompt_id in seen:
            raise DataException(f"{where}: duplicate prompt id {prompt_id!r}")
        seen.add(prompt_id)

        prompts.append(
            Prompt(
                id=prompt_id,
                task="gsm8k",
                turns=[question],
                reference=_gsm8k_final_answer(answer),
                meta={
                    "source": "openai/gsm8k",
                    "split": split,
                    "index": row_index,
                },
            )
        )

    return prompts


def _ensure_jsonl(source: DatasetSource, downloader: Any) -> Path:
    """Вернуть путь к JSONL в кэше; при отсутствии — скачать, проверить и переименовать."""
    cache_dir = _cache_root() / "datasets" / source.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{source.split or source.name}.jsonl"

    if path.exists():
        _verify_sha256(path, source.sha256)
        return path

    tmp = path.with_suffix(".part")
    downloader(source, tmp)
    _verify_sha256(tmp, source.sha256)
    tmp.replace(path)
    return path


def _download_spec_bench(source: DatasetSource, target: Path) -> None:
    url = source.location.format(revision=source.revision or "main")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            with open(target, "wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
    except httpx.HTTPError as exc:
        raise DataException(f"cannot download {url}: {exc}") from exc


def _download_gsm8k(source: DatasetSource, target: Path) -> None:
    """Забрать GSM8K постранично через datasets-server и записать канонический JSONL."""
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None

    while True:
        try:
            response = httpx.get(
                GSM8K_ROWS_URL,
                params={
                    "dataset": source.location,
                    "config": source.config or "main",
                    "split": source.split or "test",
                    "offset": offset,
                    "length": 100,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DataException(f"cannot read {source.location}: {exc}") from exc

        batch = payload.get("rows", [])
        total = payload.get("num_rows_total", total)
        if not batch:
            break

        for item in batch:
            rows.append(
                {
                    "idx": item["row_idx"],
                    "question": item["row"]["question"],
                    "answer": item["row"]["answer"],
                }
            )
        offset += len(batch)
        if total is not None and offset >= total:
            break

    if total is not None and len(rows) != total:
        raise DataException(f"gsm8k: got {len(rows)} rows, expected {total}")

    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            fh.write(line + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataException(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise DataException(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _normalize_turns(raw: object, where: str) -> list[str]:
    """Spec-Bench хранит turns как список строк; иногда встречается список dict с content."""
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list):
        raise DataException(f"{where}: turns must be a list, got {type(raw).__name__}")

    turns: list[str] = []
    for turn_index, turn in enumerate(raw):
        if isinstance(turn, str):
            text = turn
        elif isinstance(turn, dict) and isinstance(turn.get("content"), str):
            text = turn["content"]
        else:
            raise DataException(f"{where}: bad turn {turn_index}: {turn!r}")

        if not text.strip():
            raise DataException(f"{where}: empty turn {turn_index}")
        turns.append(text)

    if not turns:
        raise DataException(f"{where}: turns is empty")
    return turns


def _gsm8k_final_answer(answer: str) -> str:
    """В GSM8K финальный ответ идёт после ####; если маркера нет — возвращаем весь ответ."""
    marker = "####"
    if marker in answer:
        return answer.rsplit(marker, 1)[1].strip()
    return answer.strip()


def _required(row: dict[str, Any], key: str, where: str) -> Any:
    try:
        return row[key]
    except KeyError as exc:
        raise DataException(f"{where}: missing field {key!r}") from exc


def _select(prompts: list[Prompt], n: int | None, seed: int,
            tasks: list[str] | None) -> list[Prompt]:
    if tasks is not None:
        wanted = {task.strip().lower() for task in tasks}
        prompts = [prompt for prompt in prompts if prompt.task.lower() in wanted]
        if not prompts:
            raise DataException(f"no prompts for tasks={tasks}")

    prompts = sorted(prompts, key=lambda prompt: prompt.id)

    if n is None:
        return prompts
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise DataException("n must be a positive int or None")
    if n > len(prompts):
        raise DataException(f"n={n}, but only {len(prompts)} prompts after filtering")

    ranked = sorted(prompts, key=lambda prompt: _stable_rank(prompt.id, seed))
    return ranked[:n]


def _stable_rank(prompt_id: str, seed: int) -> bytes:
    """Ключ сортировки, не зависящий от PYTHONHASHSEED и версии Python."""
    return hashlib.sha256(f"{seed}:{prompt_id}".encode("utf-8")).digest()


def _cache_root() -> Path:
    env = os.environ.get("FASTINFER_CACHE")
    if env:
        return Path(env).expanduser()
    if Path("/kaggle").exists():
        return Path("/kaggle/temp/fastinfer")
    return Path.home() / ".cache" / "fastinfer"


def _verify_sha256(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    actual = _sha256(path)
    if actual != expected:
        raise DataException(
            f"{path}: sha256 {actual} != pinned {expected}; "
            "delete the cache file and check the dataset revision"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_turn(prompt: Prompt, turn: int, history: list[Message],
                spec: ModelSpec, gen: GenParams) -> list[Message]:
    """Собрать сообщения для одного хода: system? + history + текущая реплика.

    `history` — уже готовые сообщения предыдущих ходов (user/assistant). Для первого
    хода передаётся пустой список. Рассуждение включается/выключается под модель:
    для `system:/think|/no_think` — системным сообщением, для `template:...` —
    через `template_kwargs` на стороне движка.
    """
    if not 0 <= turn < len(prompt.turns):
        raise DataException(
            f"{prompt.id}: turn {turn} out of range 0..{len(prompt.turns) - 1}"
        )
    if gen.reasoning not in {"on", "off"}:
        raise DataException(f"gen.reasoning must be 'on' or 'off', got {gen.reasoning!r}")

    messages: list[Message] = []
    system = _system_message(spec, gen)
    if system is not None:
        messages.append(system)
    messages.extend(history)
    messages.append({"role": "user", "content": prompt.turns[turn]})
    return messages


def template_kwargs(spec: ModelSpec, gen: GenParams) -> dict[str, bool]:
    """Параметры chat template для режима рассуждения.

    Для моделей с `template:enable_thinking` движок передаёт их в /apply-template
    или /v1/chat/completions; для `system:...` ничего не нужно.
    """
    switch = spec.reasoning_switch

    if switch == "system:/think|/no_think":
        return {}
    if switch.startswith("template:"):
        key = switch.split(":", 1)[1].strip()
        if not key:
            raise DataException(f"{spec.name}: empty template reasoning switch")
        return {key: gen.reasoning == "on"}
    raise DataException(f"{spec.name}: unknown reasoning switch {switch!r}")


def _system_message(spec: ModelSpec, gen: GenParams) -> Message | None:
    switch = spec.reasoning_switch
    if switch == "system:/think|/no_think":
        return {"role": "system", "content": "/think" if gen.reasoning == "on" else "/no_think"}
    if switch.startswith("template:"):
        return None
    raise DataException(f"{spec.name}: unknown reasoning switch {switch!r}")

# Ядро `fastinfer`: контракт и план

Это план и сигнатуры. Реализацию пишет разработчик ядра: он хочет разобраться сам, поэтому в заготовке
репозитория есть только контракт — `types.py`, `engines/base.py`, функции в `__init__.py` с
`raise NotImplementedError`, шаблоны конфигов и пример результата. Всё остальное — его решения.

Контракт (форматы, типы, интерфейс движка, функции для исследователя) общий для всей команды:
менять его можно только через PR, который смотрят все трое. Внутренности модулей — свободны.

## 1. Принципы

- **Ядро — библиотека внутри одного запуска.** На Kaggle, ноутбуке, Mac или ПК оно берёт конфиг
  и пишет папку результата. Об очереди, Kaggle API, GitHub, Hugging Face и Telegram оно не знает —
  это инфраструктура (`docs/infra.md`).
- **Общий API — это два формата**: `config.yaml` на входе, папка результата на выходе. Исследователь пишет
  конфиг, ядро исполняет, диспетчер читает результат.
- **Один код на всех устройствах.** Три движка с одним интерфейсом: llama.cpp (основной),
  HF transformers (прототипы и анализ), API Nebius (Super, качество MoE).
- **База и варианты — в одном запуске**, на одном устройстве; повторы чередуются.
- **Всё записывается**: коммит, конфиг, флаги, версии, устройство, потолок.
- **Честный замер встроен в ядро**, а не остаётся на совести эксперимента (раздел 9).

## 2. Структура пакета

```
core/
  pyproject.toml          # пакет fastinfer: pip install -e core; extras [hf], [api], [dev]
  fastinfer/
    __init__.py           # функции для исследователя (раздел 7.1)
    types.py              # типы контракта (раздел 5)
    config.py             # чтение и проверка config.yaml, варианты, перебор
    results.py            # запись и чтение папки результата, heartbeat, сводная строка
    models.py             # реестр моделей: файлы, кванты, архитектура, выборка, имя в API
    hardware.py           # описание устройства, ПСП, потолки
    data.py               # наборы промптов, оформление под модель (режим рассуждения)
    engines/
      base.py             # интерфейс Engine (раздел 6)
      llamacpp.py         # llama-server, llama-bench, квантизация, перплексия
      hf.py               # transformers: генерация, логиты, хуки, снимок состояния Mamba
      api.py              # Nebius: chat, completions, учёт денег
      fake.py             # поддельный движок для тестов без моделей
    measure.py            # прогрев, замер скорости и качества, A/B с чередованием
    compare.py            # ускорение, интервалы, совпадение с базой, вердикт
    quality.py            # разбор и проверка ответов по задачам
    spec.py               # формулы спекуляции, офлайн-приём
    analysis.py           # разреженность активаций, важность слоёв, эксперты
    run.py                # python -m fastinfer.run
    selftest.py           # python -m fastinfer.selftest
  templates/              # готовые конфиги под семейства
  tests/
```

## 3. Формат `config.yaml`

| Поле | Тип | Обязательно | Что значит |
|---|---|---|---|
| `name` | str | да | короткое имя латиницей через дефис |
| `family` | int 1–12 | да | номер семейства из `docs/families.md` |
| `hypothesis` | str | да | что проверяем и чего ждём |
| `device` | str | да | `kaggle-t4x2`, `laptop-780m`, `mac`, `pc-cpu` или `any`; ядро сверяет с настоящим устройством |
| `mode` | str | да | `speed`, `quality`, `offline` или `analysis` (ниже) |
| `model` | str | да* | имя из реестра; можно переопределить в варианте |
| `quant` | str | нет | для llamacpp — квант GGUF (`Q4_K_M`…), для hf — dtype (`fp16`, `bf16`, `int8`, `nf4`) |
| `dataset` | map | да | `name`, `n`, `seed` (0), `tasks` (список или null) |
| `gen` | map | нет | поля `GenParams` (раздел 5) |
| `repeats` | int | нет | 3; в `quality` всегда 1 |
| `budget_usd` | float | нет | потолок денег API на запуск; 0 — API запрещён |
| `timeout_min` | int | нет | 600; ядро само заканчивает работу и пишет то, что успело |
| `base` | VariantSpec | да | вариант-база |
| `variants` | list[VariantSpec] | да | у каждого `name`; может быть пустым в `analysis` |
| `notes` | str | нет | любые пояснения |

**VariantSpec**: `name`, `engine` (`llamacpp`, `hf`, `api`), `model`, `quant`, `flags` (map),
`gen` (частичные `GenParams`), `analysis` (map, для `mode: analysis`), `meta` (map; например,
`draft_cost` — цена шага черновика относительно цели для офлайн-оценки), `sweep` (map).
Незаданные поля варианта берутся из базы, а у базы — из верхнего уровня конфига.

**sweep** размножает вариант: `{spec_draft_n_max: [1, 2, 4]}` даёт три варианта с именами
`<name>-spec_draft_n_max=1` и т. д. Ключ без префикса — флаг движка; `gen.<поле>` — параметр генерации;
`model` и `quant` — модель и квант. Несколько ключей перемножаются.

**Режимы**:
- `speed` — батч 1, жадный выбор, повторы с чередованием; результат — скорость и совпадение с базой.
- `quality` — генерация батчем (параллельные слоты), проверка ответов; результат — точность, токены, деньги.
- `offline` — без замера времени: база — цель (например, Super через API), варианты — черновики;
  результат — доля совпадений и τ по γ, расчётное ускорение по формуле.
- `analysis` — проход по модели с хуками (HF); результат — статистики (`analysis` в варианте задаёт какие).

**Флаги llama.cpp** — ключи без дефисов; ядро переводит их в аргументы и падает на неизвестном ключе
до загрузки модели:

| Ключ | Аргумент | Пример |
|---|---|---|
| `ngl` | `-ngl` | 99 |
| `fa` | `-fa` | on, off, auto |
| `sm`, `ts` | `-sm`, `-ts` | layer, "1,1" |
| `t`, `tb` | `-t`, `-tb` | 4 |
| `ub`, `ctx`, `np` | `-ub`, `-c`, `-np` | 512, 8192, 4 |
| `ctk`, `ctv` | `-ctk`, `-ctv` | q8_0 |
| `spec_type` | `--spec-type` | ngram-mod, ngram-cache, draft-simple, draft-mtp, draft-eagle3, draft-dflash |
| `model_draft` | `-md` | имя из реестра, например `nano-4b:Q4_K_M` → путь к файлу |
| `ngld` | `-ngld` | 99 |
| `spec_draft_n_max`, `spec_draft_n_min`, `spec_draft_p_min` | `--spec-draft-n-max` … | 4, 0, 0.0 |
| `spec_draft_device` | `--spec-draft-device` | CUDA1 |
| `reasoning_budget` | `--reasoning-budget` | 512 (значение сервера по умолчанию) |
| `override_kv` | `--override-kv` | `nemotron_h_moe.expert_used_count=int:4` |
| `n_cpu_moe` | `--n-cpu-moe` | 10 |
| `extra` | как есть | список строк для всего остального |

Пример — `core/templates/spec_compare.yaml`. В YAML значения `on` и `off` пишем в кавычках: иначе PyYAML
прочитает их как `true` и `false`. У `llama-server` с `np: N` контекст `ctx` делится между слотами поровну.

**Начальный реестр моделей** (`models.py`; имена используются в конфигах и в `model_draft`):

| Имя | Модель | Исходные веса | GGUF | В API Nebius |
|---|---|---|---|---|
| `nano-4b` | Nemotron 3 Nano 4B | уточнить имя репозитория | `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` | — |
| `nano-9b-v2` | Nemotron Nano 9B v2 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | свой из исходных весов (`convert_hf_to_gguf.py` из llama.cpp) или выбрать готовый и закрепить ревизию | — |
| `nano-30b-a3b` | Nemotron 3 Nano 30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` | имя из `GET /v1/models` |
| `lightning-3.5` | Lightning 3.5 (30B-A3B) | уточнить | уточнить; нужен GGUF с MTP-слоями | имя из `GET /v1/models` |
| `super-3` | Nemotron 3 Super 120B-A12B | — | — | имя из `GET /v1/models` |
| `super-3.5-vl` | Nemotron 3.5 Super VL | — | — | имя из `GET /v1/models` |

Ревизию (коммит) каждого репозитория HF закрепляем в реестре, sha256 файлов пишем в `env.json`.

## 4. Папка результата

```
results/<job_id>/
  result.json       # сводка: варианты, скорость, качество, сравнения, вердикты, деньги, предупреждения
  per_prompt.jsonl  # одна строка на (вариант, повтор, промпт, ход)
  env.json          # разрешённый конфиг, аргументы движков, версии, устройство, файлы моделей и sha256, потолок
  logs/             # логи серверов llama.cpp и самого ядра
  heartbeat.json    # «я жив»: время, этап, прогресс 0–1 — обновляется раз в минуту
```

Строка `per_prompt.jsonl`: `variant`, `repeat`, `prompt_id`, `task`, `turn`, `n_prompt`, `n_gen`,
`n_thinking`, `n_answer`, `ttft_ms`, `decode_ms`, `total_ms`, `tok_s`, `draft_n`, `draft_accepted`,
`finish_reason`, `correct` (в `quality`), `cost_usd`, `text`; `token_ids` и `top2_margin` — только
в первом повторе (для проверки совпадения).

`result.json` — это `RunResult` (раздел 5): у вариантов нет списка генераций (он в `per_prompt.jsonl`),
полный разрешённый конфиг лежит в `env.json`. Пример — `docs/result_example.json`. Поле `schema` — номер
версии формата; при несовместимом изменении увеличиваем его.

## 5. Типы (`types.py`)

```python
from dataclasses import dataclass, field

@dataclass
class Prompt:
    id: str                          # "spec_bench/qa/81"
    task: str                        # "qa", "gsm8k", "math500", ...
    turns: list[str]                 # реплики пользователя; у MT-bench их две
    reference: str | None = None     # эталонный ответ для проверки качества
    meta: dict = field(default_factory=dict)

@dataclass
class GenParams:
    max_tokens: int = 256
    temperature: float = 0.0         # 0 — жадный выбор
    top_p: float = 1.0
    seed: int = 0
    reasoning: str = "off"           # "off" | "on"
    reasoning_budget: int = -1       # −1 без ограничений, 0 — сразу закончить, N — бюджет в токенах
    logprobs: int = 0                # сколько лучших вероятностей вернуть на токен, 0–20
    stop: list[str] = field(default_factory=list)

@dataclass
class Generation:                    # один ход одного промпта на одном движке
    prompt_id: str
    turn: int
    text: str                        # весь текст, включая рассуждение
    token_ids: list[int]
    n_prompt: int
    n_gen: int
    n_thinking: int
    n_answer: int
    ttft_ms: float                   # до первого токена
    decode_ms: float                 # генерация без prefill
    total_ms: float                  # от запроса до последнего токена
    tok_s: float                     # скорость генерации без prefill
    draft_n: int = 0                 # предложено токенов черновиком
    draft_accepted: int = 0          # из них принято
    logprobs: list | None = None     # по позициям: [(token_id, logprob), ...] лучших k
    finish_reason: str = ""
    cost_usd: float = 0.0            # только API

@dataclass
class SpeedStats:
    tok_s_median: float
    tok_s_p10: float
    tok_s_p90: float
    ttft_ms_median: float
    total_s_median: float            # время до ответа
    acceptance: float | None         # draft_accepted / draft_n
    tokens_per_pass: float | None    # τ
    frac_of_ceiling: float | None    # tok_s × байты на токен ÷ ПСП

@dataclass
class QualityStats:
    accuracy: float
    ci95: tuple[float, float]        # Уилсон
    n: int
    mean_thinking_tokens: float
    mean_answer_tokens: float
    cost_usd: float

@dataclass
class Report:                        # всё про один вариант
    variant: str
    mode: str
    generations: list[Generation]    # все повторы
    speed: SpeedStats | None
    quality: QualityStats | None
    engine_info: dict                # версия, аргументы, устройство

@dataclass
class Divergence:
    prompt_id: str
    position: int                    # номер токена, где ответ разошёлся с базой
    base_margin: float               # отрыв logprob лучшего токена от второго у базы

@dataclass
class Comparison:
    variant: str
    base: str
    speedup: float | None            # медиана отношений tok_s по одним и тем же промптам
    speedup_ci95: tuple[float, float] | None
    total_time_ratio: float | None   # время до ответа: база ÷ вариант
    exact_match: float | None        # доля промптов, где ответ совпал с базой
    divergences: list[Divergence]
    numeric_divergences: int         # из них «почти ничья» у базы — похоже на округление
    accuracy_delta: float | None     # п.п., вариант − база
    accuracy_delta_ci95: tuple[float, float] | None
    tokens_ratio: float | None       # средние токены: вариант ÷ база
    cost_ratio: float | None
    verdict: str                     # «быстрее, без потерь», «быстрее, точность −2 п.п.», «не значимо», «медленнее»
    warnings: list[str] = field(default_factory=list)

@dataclass
class AcceptStats:                   # офлайн-приём черновика
    alpha: float                     # доля позиций, где черновик угадал токен цели
    tau_by_gamma: dict[int, float]   # среднее число токенов за проход цели при длине черновика γ
    accepted_len_hist: dict[int, int]
    predicted_speedup_by_gamma: dict[int, float] | None   # если задан draft_cost
    n_tokens: int

@dataclass
class DeviceInfo:
    kind: str                        # kaggle-t4x2, laptop-780m, mac, pc-cpu
    backend: str                     # cuda, vulkan, metal, cpu
    gpus: list[dict]                 # name, vram_gb, compute_capability
    cpu: str
    cores: int
    ram_gb: float
    os: str
    bandwidth_gbs: float             # пиковая ПСП из таблицы или замера
    versions: dict                   # python, fastinfer, llama.cpp, torch, transformers, драйвер

@dataclass
class ModelSpec:
    name: str                        # "nano-9b-v2"
    kind: str                        # "dense" | "moe"
    hf_repo: str | None              # исходные веса
    gguf_repo: str | None
    gguf_files: dict[str, str]       # квант → имя файла
    arch: dict                       # слои по типам, d_model, словарь, эксперты
    params_per_token: float          # читается на токен, без входного эмбеддинга
    sampling: dict                   # рекомендованная выборка с рассуждением и без
    reasoning_switch: str            # "system:/think|/no_think" | "template:enable_thinking"
    api_id: str | None               # имя модели в Nebius
    license: str

@dataclass
class DatasetSpec:
    name: str
    n: int | None = None
    seed: int = 0
    tasks: list[str] | None = None

@dataclass
class VariantSpec:
    name: str
    engine: str
    model: str | None = None
    quant: str | None = None
    flags: dict = field(default_factory=dict)
    gen: dict = field(default_factory=dict)
    analysis: dict | None = None
    meta: dict = field(default_factory=dict)
    sweep: dict = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    name: str
    family: int
    hypothesis: str
    device: str
    mode: str
    model: str | None
    quant: str | None
    dataset: DatasetSpec
    gen: GenParams
    base: VariantSpec
    variants: list[VariantSpec]
    repeats: int = 3
    budget_usd: float = 0.0
    timeout_min: int = 600
    notes: str = ""

@dataclass
class RunResult:                     # то, что пишется в result.json
    schema: int
    job_id: str
    commit: str
    name: str
    family: int
    mode: str
    device: DeviceInfo
    started_at: str
    finished_at: str
    ceiling_tok_s: float | None
    variants: dict[str, Report]      # в result.json — без generations (они в per_prompt.jsonl)
    comparison: dict[str, Comparison]
    accept: dict[str, AcceptStats] = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)
```

## 6. Интерфейс движка (`engines/base.py`)

```python
from typing import Protocol

class Engine(Protocol):
    name: str
    def start(self) -> None: ...                 # загрузить модель или поднять сервер
    def stop(self) -> None: ...
    def generate(self, prompt: Prompt, gen: GenParams) -> list[Generation]: ...   # по одной на ход
    def generate_many(self, prompts: list[Prompt], gen: GenParams,
                      parallel: int = 1) -> list[list[Generation]]: ...
    def tokenize(self, text: str) -> list[int]: ...
    def info(self) -> dict: ...                  # версия, аргументы, устройство
```

`load_model` возвращает движок как контекстный менеджер: `with load_model(...) as m:` — `start` и `stop`
вызываются сами.

## 7. Функции

Цифра в комментарии — этап (раздел 12): 0 — нужно первым, 1 — следом, 2 — позже.

### 7.1. Для исследователя (`__init__.py`)

```python
load_config(path) -> ExperimentConfig                                        # 0
run_experiment(cfg: ExperimentConfig, out_dir) -> RunResult                  # 0  весь конфиг целиком
load_model(name, engine="llamacpp", quant=None, flags=None) -> Engine        # 0
load_prompts(dataset, n=None, seed=0, tasks=None) -> list[Prompt]            # 0
generate(engine, prompts, gen, parallel=1) -> list[list[Generation]]         # 0
measure(engine, prompts, gen, mode="speed", repeats=3) -> Report             # 0 speed, 1 quality
compare(base: Report, variant: Report) -> Comparison                         # 0
save_results(result: RunResult, out_dir) -> dict[str, Path]                  # 0
device_info() -> DeviceInfo                                                  # 0
ceiling(model, quant, device=None) -> float                                  # 0  потолок ток/с
```

### 7.2. Конфиг и результат (`config.py`, `results.py`)

```python
validate_config(raw: dict) -> list[str]           # 0  все ошибки сразу, до загрузки моделей
expand_variants(cfg) -> list[VariantSpec]         # 0  база + варианты + перебор
config_hash(cfg) -> str                           # 0
load_results(path) -> RunResult                   # 0
summary_line(result) -> str                       # 0  одна строка для Telegram
heartbeat(out_dir, stage: str, progress: float)   # 0
```

### 7.3. Модели и устройство (`models.py`, `hardware.py`)

```python
resolve_model(name, quant=None) -> ModelSpec      # 0
fetch_model(spec, quant, cache_dir) -> Path       # 0  скачать с HF, проверить sha256
bytes_per_token(spec, quant) -> float             # 0
peak_bandwidth(device) -> float                   # 0  таблица: T4 — 320, 780M — по типу памяти, M1 — 68 …
measure_bandwidth() -> float                      # 2
```

### 7.4. Данные и качество (`data.py`, `quality.py`)

```python
load_prompts("spec_bench" | "gsm8k" | "math500" | "aime", ...)   # 0: spec_bench, gsm8k; 1: остальные
render_turn(prompt, turn, history, spec, gen) -> list[dict]      # 0  включает или выключает рассуждение под модель
split_thinking(text) -> tuple[str, str]                          # 1  рассуждение и ответ
extract_answer(text, task) -> str | None                         # 1  \boxed{}, последнее число, буква варианта
score(prompt, generation) -> float                               # 1  0 или 1
```

### 7.5. Движки (`engines/`)

```python
# llamacpp.py
class LlamaCppServer(Engine)                                     # 0  llama-server: запуск, /health, запросы, timings
flags_to_argv(flags: dict) -> list[str]                          # 0
llama_bench(model, flags, n_prompt=512, n_gen=128, reps=5) -> dict   # 0  -o json
batched_bench(model, flags, npp, ntg, npl) -> list[dict]             # 1
perplexity(model, text_path, flags) -> float                         # 1
make_imatrix(model, calib_path, out_path, flags) -> Path             # 1
quantize(src, dst, qtype, imatrix=None, tensor_types=None,
         output_type=None, embedding_type=None) -> Path              # 1  tensor_types: {регулярка: тип}

# api.py
class ApiEngine(Engine)                                          # 1  OpenAI-совместимый клиент Nebius
ApiEngine.score_text(context, continuation, k=1) -> list         # 1  completions + echo + logprobs: вероятности готового текста
class Ledger                                                     # 1  учёт денег
    Ledger.estimate(model, n_req, in_tok, out_tok, batch=False) -> float
    Ledger.charge(model, usage) -> float                         # BudgetExceeded при превышении потолка

# hf.py
class HFEngine(Engine)                                           # 1  прототипы; скорость отсюда не публикуем
HFEngine.logits(input_ids) -> Tensor                             # 1  один проход по готовому тексту
HFEngine.add_hook(where, fn) -> Handle                           # 2  "mlp.act", "router", "mamba.in_proj"
class MambaState                                                 # 2  snapshot(cache), restore(cache)

# fake.py
class FakeEngine(Engine)                                         # 0  детерминированный, для тестов
```

### 7.6. Замер и сравнение (`measure.py`, `compare.py`)

```python
warmup(engine, prompts, gen, n=2)                                # 0
measure_speed(engine, prompts, gen, repeats=3) -> Report         # 0
measure_quality(engine, prompts, gen, parallel=8) -> Report      # 1
run_ab(variants, prompts, gen, mode, repeats, interleave=True) -> dict[str, Report]   # 0
timer(sync_cuda=True)                                            # 0  контекст для своего кода
exact_match(base, var) -> tuple[float, list[Divergence]]         # 0
bootstrap_ci(values, n_boot=2000, seed=0) -> tuple[float, float] # 0
wilson_ci(k, n) -> tuple[float, float]                           # 1
pareto(points, x="mean_tokens", y="accuracy") -> list[dict]      # 1  кривые «точность — токены»
verdict(cmp: Comparison) -> str                                  # 0
```

### 7.7. Спекуляция (`spec.py`)

```python
expected_tokens(alpha, gamma) -> float                           # 0  (1 − α^(γ+1)) / (1 − α)
speedup(alpha, gamma, c, v=1.05) -> float                        # 0  E / (γ·c + v)
best_gamma(alpha, c, v=1.05, max_gamma=8) -> tuple[int, float]   # 0
greedy_acceptance(target_ids, draft_argmax, gammas=range(1, 9)) -> AcceptStats   # 1
ngram_proposer(n=3, m=4) -> Callable                             # 1  черновик для офлайн-симуляции
moe_verify_cost(n_tokens, n_experts=128, k=6, routed_share=0.39) -> float          # 1
tokenizer_compat(model_a, model_b, texts) -> dict                # 1  годится ли модель в черновики
```

`greedy_acceptance` в жадном режиме точен: черновик прогоняется «с учителем» по готовому ответу цели
(`HFEngine.logits` или `ApiEngine.score_text`), и принятая длина с позиции i — это число подряд идущих
совпадений его argmax с токенами цели, не больше γ. Нужен одинаковый токенизатор.

### 7.8. Анализ (`analysis.py`, этап 2)

```python
activation_sparsity(engine, prompts, block_sizes=(32, 256)) -> dict   # доля точных нулей ReLU² по слоям и блокам
layer_importance(engine, prompts, metric="ppl") -> list[dict]         # что будет, если убрать слой
expert_usage(engine, prompts) -> dict      # частоты экспертов и общие эксперты у соседних токенов
```

## 8. Детали движков

**llama.cpp** (сверено с исходниками сборки от 15.10.2026, `tools/server`):
- Без рассуждения удобно: `POST /apply-template` → `POST /completion` с `cache_prompt: false`,
  `return_tokens: true`, `n_probs` (для отрыва top-2), `temperature`, `seed`, `n_predict`.
- С бюджетом рассуждения: `POST /v1/chat/completions` с `reasoning_budget_tokens`,
  `chat_template_kwargs: {"enable_thinking": ...}` и `reasoning_format: "none"` (сырой текст с `<think>`).
  Бюджет меняется в каждом запросе — сервер перезапускать не нужно.
- `timings` в ответе: `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `predicted_per_second`,
  `cache_n`, а при черновике — `draft_n` и `draft_n_accepted`.
- Готовность — `GET /health`; параметры модели — `GET /props`; токены — `POST /tokenize`.
- Бинарники берутся из `LLAMA_CPP_BIN`; ядро их не собирает.

**HF transformers 5.19.0**: нативный NemotronH. Встроенный assisted generation для моделей с состоянием
не работает — спекуляция в HF только своим циклом с откатом (`MambaState`, основа —
`experiments/2026-10-15-mamba-rollback-check/rollback_check.py`). На T4 — только fp16; 9B в fp16 (~18 ГБ)
занимает обе карты (`device_map="auto"`).

**API Nebius**: OpenAI-совместимый, `https://api.tokenfactory.nebius.com/v1/`, ключ `NEBIUS_API_KEY`.
`chat`: `logprobs`/`top_logprobs` до 20, `reasoning_effort`, `service_tier: flex`; `max_tokens`
по умолчанию 8192 — задавать явно. `completions`: `echo` и `logprobs` до 20 — для оценки готового текста.
Batch API — на 50% дешевле, ответ в течение суток. Лимиты из примера в документации — 60 запросов
и 400 тыс. токенов в минуту. Каждый ответ кэшируется (одинаковый запрос не оплачиваем дважды),
каждый расход пишется в `Ledger`. Имена моделей — из `GET /v1/models`.

## 9. Правила замера

1. Прогрев на двух промптах, в счёт не идёт.
2. Для скорости: батч 1, `temperature: 0`, одинаковые промпты и `max_tokens`, `cache_prompt: false`.
3. Не меньше трёх повторов, порядок вариантов чередуется (A B C, B C A, C A B). Если варианты помещаются
   в память вместе, серверы держим поднятыми; иначе — перезапуск на каждый блок.
4. Скорость генерации — без prefill (`predicted_per_second`); отдельно TTFT и время до ответа.
5. Ускорение — медиана отношений по одним и тем же промптам, интервал — бутстрепом.
6. «Без потерь» — сравнение ответов с базой в жадном режиме. Небольшие расхождения бывают и при правильном
   коде: путь «по одному токену» и путь проверки считают по-разному (в fp32 ~1e-3 по логитам,
   см. rollback_check). Поэтому для каждого расхождения ядро берёт отрыв logprob лучшего токена
   от второго у базы: почти ничья (меньше ~0,1) — округление, большой отрыв — ошибка.
7. Методы, которые меняют ответы, — всегда со скоростью и точностью; качество мерим батчем.
8. В `env.json` всегда: коммит, конфиг, аргументы, версии, устройство, файлы моделей, потолок.
9. На Kaggle не мерим скорость параллельно с другой задачей на той же машине.

## 10. Что ядро ждёт от инфраструктуры

| Переменная | Что в ней |
|---|---|
| `LLAMA_CPP_BIN` | папка с llama-server, llama-bench, llama-batched-bench, llama-quantize, llama-imatrix, llama-perplexity |
| `FASTINFER_CACHE` | кэш весов вне папки результата; на Kaggle — не в /kaggle/working |
| `FASTINFER_JOB_ID`, `FASTINFER_COMMIT` | номер задачи и коммит (или флаги командной строки) |
| `FASTINFER_DEVICE` | тип устройства, если его нельзя надёжно определить |
| `FASTINFER_API_CAP_USD` | потолок денег API на задачу; меньшее из этого и `budget_usd` |
| `NEBIUS_API_KEY`, `HF_TOKEN` | ключи; только из секретов, никогда в логах |

## 11. Командная строка

```
python -m fastinfer.run CONFIG --out DIR [--job-id N] [--commit SHA] [--dry-run]
python -m fastinfer.selftest          # весь путь на FakeEngine за секунды, без сети
python -m fastinfer.ceiling nano-9b-v2 Q4_K_M [--device auto]
```

`--dry-run` проверяет конфиг, находит модели, оценивает время и деньги — без запуска.

| Код выхода | Значение |
|---|---|
| 0 | готово |
| 2 | ошибка в конфиге |
| 3 | не то устройство |
| 4 | кончился бюджет API |
| 5 | упал движок |
| 6 | вышел `timeout_min` (результат частичный) |

## 12. Этапы

| Этап | Что входит | Какие семейства открывает |
|---|---|---|
| 0 | типы и контракт; конфиг и результат; реестр моделей; устройство и потолки; spec_bench и gsm8k; llama.cpp; FakeEngine; замер скорости; A/B; сравнение и вердикт; формулы спекуляции; командная строка; selftest | 1 (n-граммы, черновик 4B, MTP), 3 (скорость квантов), 7, 9 |
| 1 | качество и проверка ответов; бюджет рассуждения; API и учёт денег; офлайн-приём; квантизация, imatrix, перплексия; совместимость токенизаторов; logits в HF | 2, 3 (качество), 4 (число экспертов), 5, 1 для Super |
| 2 | хуки HF; MambaState; анализ; замер ПСП | 6, 8, 4 (общие эксперты), само-черновик из 1 |

Этап 0 готов, когда:
- тесты на FakeEngine проходят в CI на GitHub;
- на ноутбуке `python -m fastinfer.run core/templates/spec_compare.yaml --out results/local` с 4B в Q4
  даёт полную папку результата;
- тот же конфиг без правок проходит на Kaggle 2× T4 через launch.py (`docs/infra.md`).

## 13. Как это выглядит у исследователя

Обычно хватает конфига (`core/templates/`): чередование повторов и запись результата ядро делает само.
Код нужен, когда идея в конфиг не укладывается:

```python
from fastinfer import load_model, load_prompts, measure, compare, GenParams

prompts = load_prompts("spec_bench", n=120, tasks=["summarization", "rag"])
gen = GenParams(max_tokens=256)                     # жадно, без рассуждения

with load_model("nano-9b-v2", quant="Q4_K_M", flags={"ngl": 99}) as m:
    base = measure(m, prompts, gen)
with load_model("nano-9b-v2", quant="Q4_K_M", flags={"ngl": 99, "spec_type": "ngram-mod"}) as m:
    var = measure(m, prompts, gen)

print(compare(base, var).verdict)
```

## 14. Как менять контракт

1. PR с изменением `types.py`, `engines/base.py`, сигнатур в `__init__.py` или форматов.
2. В описании PR — что меняется и кого затронет (исследователь, диспетчер).
3. Одобряют все трое; при несовместимом изменении формата результата — `schema` + 1.

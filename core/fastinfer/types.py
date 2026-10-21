"""Типы контракта ядра — docs/core_api.md, раздел 5.

Только данные, без логики. Это общий контракт команды: менять — только через PR, который смотрят все трое.
"""

from dataclasses import dataclass, field



def todo(*args):
    pass

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

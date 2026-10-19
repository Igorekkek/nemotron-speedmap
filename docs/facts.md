# Проверенные факты

Факты, которые мы уже проверили, — чтобы не искать их заново и не путать. У каждого раздела указан источник.
Пометка «перепроверить» — данные из ранних заметок без сохранённой ссылки или из вторичных источников.
Расчёты (байты на токен, потолки) — наши, по архитектуре; метод — в `docs/families.md`.

## Модели

### Nemotron Nano 9B v2 — `nvidia/NVIDIA-Nemotron-Nano-9B-v2`

- 56 слоёв: 27 Mamba-2, 4 внимания, 25 MLP. Схема:
  `M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-` (M — Mamba-2, * — внимание, - — MLP).
- d_model 4480, FFN 15 680, 40 голов внимания, 8 групп KV, Mamba: 128 голов × 80, состояние 128.
- MLP — ReLU² без гейта. Словарь 131 072.
- Получена из 12B обрезкой и дистилляцией (Minitron): слои удаляли по важности (MSE); обрезка голов Mamba
  «сильно била по точности».
- Рассуждение по умолчанию включено; `/think` и `/no_think` в системном промпте или в реплике.
  Бюджет рассуждения: после бюджета модель дописывает строку и ставит `</think>`, принудительно —
  через бюджет + 500 токенов. ~5% обучающих данных — обрезанные рассуждения.
- Выборка: с рассуждением T = 0,6, top-p = 0,95; без рассуждения — жадно.
- В vLLM рекомендуют `--mamba_ssm_cache_dtype float32` «для точности».
- Лицензия: NVIDIA Open Model License.

Источники: [Megatron-Bridge](https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/apidocs/bridge/bridge.models.nemotronh.nemotron_h_provider.html),
[карточка](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2), [статья Nemotron Nano 2](https://arxiv.org/html/2508.14444v2).

### Nemotron 3 Nano 4B

- 42 слоя: 21 Mamba, 4 внимания, 17 MLP; 96 голов Mamba; FFN 12 544; d_model 3136.
- Получена из Nano 9B v2 (Nemotron Elastic): обрезка и дистилляция с замороженной 9B в роли учителя,
  63 млрд токенов при контексте 8K и 150 млрд при 49K.
- GGUF: `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`, `unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF`.
  Q4_K_M на Jetson Orin Nano 8 ГБ — 18 ток/с.
- Точное имя репозитория исходных весов — перепроверить.

Источники: [блог NVIDIA на HF](https://huggingface.co/blog/nvidia/nemotron-3-nano-4b),
[гайд Unsloth](https://unsloth.ai/docs/models/nemotron-3).

### Nemotron 3 Nano 30B-A3B — `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

- 52 слоя: 23 Mamba-2, 23 MoE, 6 внимания (GQA, 2 группы KV, 32 головы, head_dim 128).
- MoE: 128 маршрутизируемых экспертов + 1 общий, на токен 6; размер эксперта 1856; роутер с сигмоидой;
  ReLU² в экспертах. d_model 2688. Mamba-2: состояние 128, 8 групп, 64 головы × 64.
- Параметры: в статье 31,6 млрд всего и 3,2 млрд активных (3,6 млрд с эмбеддингами);
  в карточке — 30 млрд и 3,5 млрд активных.
- Рассуждение: `enable_thinking` в шаблоне; бюджет `reasoning_budget` (как у 9B).
  Выборка: рассуждение T = 1,0, top-p = 1,0; вызов инструментов T = 0,6, top-p = 0,95; без рассуждения — жадно.
- GGUF: `unsloth/Nemotron-3-Nano-30B-A3B-GGUF`; по Unsloth 4 бита — ~24 ГБ RAM, 8 бит — ~36 ГБ.

Источники: [карточка](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16),
[статья Nemotron 3 Nano](https://arxiv.org/html/2512.20848v1), [гайд Unsloth](https://unsloth.ai/docs/models/nemotron-3).

### Lightning 3.5 (30B-A3B), Super — перепроверить

- Lightning 3.5: из Nano 3 Base, дистиллирована на логитах Super 3 Base (top-128); есть MTP.
- Nemotron 3 Super 120B-A12B (вышла 11.03.2026): 120,6 млрд параметров, 12,7 млрд активных; 88 слоёв; d_model 4096;
  512 экспертов, на токен 22; латентная MoE 1024; 2 общих слоя MTP; словарь 131 072 (метаданные GGUF).
  Приём MTP при черновике из 7 токенов — 3,45 (SPEED-Bench), на коде 3,78, в ролевых диалогах 2,82.
  Artificial Analysis пометила Super 3 как устаревшую.
- Nemotron 3.5 Super VL (отчёт 14.10.2026), 120B-A12B: приём MTP жадно 4,287, с усилением — 4,718.

## Наши расчёты

- KV-кэш 9B: 4 слоя × 8 групп × 128 × 2 (K и V) × 2 байта = 16 КБ на токен; 134 МБ на 8 тыс. токенов.
- Состояние Mamba 9B: 27 × 128 × 80 × 128 × 4 байта ≈ 142 МБ (fp32); копия на T4 ≈ 0,4 мс.
- Байты на токен (без входного эмбеддинга): 4B — 2,14 ГБ в Q4_K_M; 9B — 5,03 ГБ; 30B-A3B — 2,12 ГБ (3,5 млрд активных).
- Выходной слой: 7% читаемого у 9B, 12% у 4B. Down-проекция MLP: 21% у 9B, 19% у 4B.
  Маршрутизируемые эксперты: ~39% читаемого у 30B-A3B.
- Потолки ток/с по устройствам — таблица в `docs/families.md`.

## Железо

| Устройство | Факты |
|---|---|
| T4 (Kaggle, две штуки) | sm75 (Turing), 16 ГБ GDDR6, ~320 ГБ/с, ~65 TFLOPS fp16 на тензорных ядрах, есть INT8 и INT4; нет bf16, FP8; FlashAttention-2, FlashInfer и Marlin требуют sm80+; карты соединены по PCIe |
| Radeon 780M (ноутбук Игоря) | встроенная, RDNA3, общая память 32 ГБ; ПСП по типу памяти: DDR5-5600 в 2 канала — 89,6 ГБ/с, LPDDR5X-7500 — 120 ГБ/с; llama.cpp через Vulkan; ROCm официально не поддерживается |
| Mac Руслана | Metal; M1 — 68 ГБ/с, M2 и M3 — 100 ГБ/с (модель Mac уточнить) |
| ПК Димы | CPU, 16 ГБ; DDR4-3200 в 2 канала — 51,2 ГБ/с |
| Сервер Игоря | ~10 ГБ диска, 1 ГБ RAM, работает постоянно |

## Kaggle

- GPU: ~30 ч в неделю на аккаунт, до 12 ч на сессию; интерактивная сессия тратит квоту, даже если простаивает.
  Квота, по всему, что удалось найти, считается по времени сессии, а не по числу карт — перепроверить по счётчику.
- CLI: `kaggle kernels push | status | output | pull | list | init`; команды остановки нет.
- `kernel-metadata.json`: `id`, `title`, `code_file`, `language`, `kernel_type`, `is_private`, `enable_gpu`,
  `enable_internet`, `machine_shape` (`NvidiaTeslaT4` — 2× T4), `dataset_sources`, `model_sources`.
- Вывод — то, что записано в `/kaggle/working`.

Источники: документация Kaggle CLI; [заметки курса Notre Dame](https://academicweb.nd.edu/~dchiang/teaching/nlp/2023/kaggle.html).

## llama.cpp — сверено с исходниками, коммит 620d222 от 15.10.2026

- Спекуляция: `--spec-type` со значениями `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`,
  `draft-dspark`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` (можно через запятую).
  По умолчанию `--spec-draft-n-max` = 3, `--spec-draft-n-min` = 0, `--spec-draft-p-min` = 0.
  `--spec-default` — это `ngram-mod` с n_match 24, n_min 48, n_max 64.
- Черновик: `-md`, `-ngld`, `--spec-draft-device` (например, `CUDA1`), `--spec-draft-n-cpu-moe`.
  Головы MTP, EAGLE-3 и DFlash скачиваются отдельным GGUF-файлом. Код EAGLE-3 явно поддерживает
  рекуррентные и гибридные цели; откат состояния есть для `nemotron_h` и `nemotron_h_moe`.
- Рассуждение: `--reasoning-budget N` (−1 — без ограничений, 0 — сразу закончить); в запросе к
  `/v1/chat/completions` — `reasoning_budget_tokens` (или `thinking_budget_tokens`),
  `chat_template_kwargs` (например, `{"enable_thinking": false}`), `reasoning_format: "none"` — сырой текст.
- Сервер `/completion`: `prompt` (текст или токены), `n_predict`, `temperature`, `top_k`, `seed`, `n_probs`,
  `cache_prompt` (по умолчанию true; может давать недетерминированность), `return_tokens`, `id_slot`.
  В ответе `timings`: `cache_n`, `prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`,
  `predicted_per_second`, при черновике — `draft_n` и `draft_n_accepted`. Ещё: `/health`, `/props`, `/tokenize`,
  `/apply-template`, `/slots`, `/metrics`.
- MoE: `--override-kv nemotron_h_moe.expert_used_count=int:4` — меньше активных экспертов без переобучения;
  `--n-cpu-moe N` — эксперты первых N слоёв в RAM.
- Прочее: `-sm none|layer|row|tensor`; `-fa on|off|auto`; квантованный V-кэш требует FlashAttention
  (сервер включает его сам); состояние рекуррентных слоёв всегда хранится в fp32.
- `llama-bench`: `-o json|jsonl|csv|md|sql`, `-r` повторы; `--override-kv` не принимает.
  `llama-batched-bench`: `-npp`, `-ntg`, `-npl 1,2,4,8,16`.
- `llama-quantize`: `--imatrix`, `--tensor-type <регулярка>=<тип>`, `--tensor-type-file`, `--output-tensor-type`,
  `--token-embedding-type`, `--prune-layers`.
- Имена тензоров: `blk.N.ssm_in`, `ssm_out`, `ssm_conv1d`, `ffn_up`, `ffn_down`, `ffn_up_exps`, `ffn_down_exps`,
  `attn_q`, `attn_output`. Активация MLP у nemotron_h — `LLM_FFN_RELU_SQR`.
- Примеры в `examples/`: `speculative`, `speculative-simple`, `lookahead`, `lookup`.

Источник: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), `tools/server/README.md`, `common/arg.cpp`,
`common/speculative.cpp`, `src/models/nemotron-h.cpp`.

## HF transformers 5.19.0

- Нативный `NemotronH`: `hybrid_override_pattern` превращается в `layers_block_type`; по умолчанию
  `mlp_hidden_act = "relu2"`; есть поля MTP (`num_nextn_predict_layers`, `mtp_layers_block_type`).
- Кэш: слои Mamba — `LinearAttentionLayer` с `conv_states` и `recurrent_states`, внимание — `DynamicLayer`
  с `keys` и `values`. На этом построен откат в `experiments/2026-10-15-mamba-rollback-check`.
- Встроенный assisted generation для моделей с состоянием не поддерживается.
- Без пакетов mamba-ssm и causal-conv1d работает медленный путь на чистом PyTorch.

## vLLM

- Бэкенды внимания: FlashAttention-2 требует sm80+; Triton и Flex помечены «любая» архитектура — на T4
  не проверено. Рецепт для 9B рассчитан на H100, H200 и B200.

Источники: [бэкенды внимания vLLM](https://docs.vllm.ai/en/latest/design/attention_backends/),
[рецепт vLLM для 9B](https://recipes.vllm.ai/nvidia/NVIDIA-Nemotron-Nano-9B-v2).

## Nebius Token Factory

- OpenAI-совместимый API: `base_url="https://api.tokenfactory.nebius.com/v1/"`, ключ `NEBIUS_API_KEY`.
- Цены за 1 млн токенов (вход / выход): Super 3 и 3.5 VL — $0,30 / $0,90; Nano 30B и Lightning — $0,06 / $0,24;
  Ultra 550B — $1 / $3. Batch API — на 50% дешевле, окно 24 часа.
- `completions`: `echo` и `logprobs` (до 20). `chat`: `logprobs` и `top_logprobs` (до 20), `service_tier: flex`,
  `reasoning_effort`; `max_tokens` по умолчанию 8192. Лимиты в примере документации — 60 запросов
  и 400 тыс. токенов в минуту.
- Скорость по Artificial Analysis: Super на Nebius ~209 ток/с, Nano 30B ~168 ток/с.
- Аренда GPU в Nebius AI Cloud в наши кредиты не входит.

Источник: [документация Nebius Token Factory](https://docs.tokenfactory.nebius.com/quickstart.md);
цены, параметры и скорости — из ранних заметок, перепроверить перед расчётами.

## Наборы данных

- Spec-Bench: 480 вопросов — MT-bench (8 категорий по 10, по две реплики), перевод, пересказ, вопросы,
  математика, RAG — по 80. Поля: `question_id`, `category`, `turns`.
  Источник: [hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench), `data/spec_bench/question.jsonl`.

## Площадки и инструменты

- Hugging Face, бесплатно: 100 ГБ приватного хранилища на пользователя или организацию; файл до 500 ГБ
  (рекомендуют меньше 200 ГБ); меньше 100 тыс. файлов в репозитории и 10 тыс. в папке.
- GitHub: в профиле можно закрепить до 6 публичных репозиториев — своих или тех, где есть вклад; файлы
  больше 100 МБ блокируются; Actions для публичных репозиториев бесплатны.
- Бесплатный Colab: запрещены воркеры распределённых вычислений, удалённое управление и несколько аккаунтов
  для обхода лимитов.
- Claude Code: Ubuntu 20.04+, Debian 10+ или Alpine 3.19+, от 4 ГБ RAM, платный план — на сервер с 1 ГБ RAM не встанет.

## Проверка отката состояния Mamba

Крошечная случайная NemotronH из 6 слоёв на CPU (`experiments/2026-10-15-mamba-rollback-check`), максимальное
расхождение логитов с эталоном:

| Проверка | fp32 | fp16 |
|---|---|---|
| A: 6 токенов одним проходом после prefill | 2,4e-6 | 3,9e-3 |
| B: снимок → чужие токены → откат → правильные | 3,3e-6 | 3,9e-3 |
| B без отката SSM-состояния | 3,2 | 3,2 |
| C: по одному токену против проверки разом | 1,6e-3 | 7,8e-3 |

Вывод: откат обязателен (без него ответ ломается), а пути «по одному токену» и «проверка разом» расходятся
численно — совпадение с базой проверяем с допуском.

# Первое задание для Claude Code: каркас репозитория

Цель — создать репозиторий `nemotron-speedmap` с каркасом и **контрактом** ядра. Реализацию ядра не писать:
её пишет разработчик ядра (`CLAUDE.md`, правило 5). Документы в `docs/`, шаблоны в `core/templates/`,
пример результата `docs/result_example.json` и эксперимент `experiments/2026-10-15-mamba-rollback-check/`
уже готовы — их не переписывать, только положить на место.

## Что создать

1. **Папки**: `core/`, `dispatcher/`, `infra/`, `experiments/`, `reports/`, `docs/`.
2. **Корень**:
   - `README.md` — что за проект (по-русски, плюс одна строка по-английски), группы моделей, ссылки
     на `docs/`, как поставить ядро и запустить тесты. Репозиторий публичный и закреплён в профилях — пусть
     README читается с первого экрана.
   - `.gitignore` — Python, виртуальные окружения, `.env`, `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`,
     локальные `results/` (кроме сводок `experiments/*/results/*.json`), кэши.
   - `.github/workflows/ci.yml` — на push и PR: Python 3.10 и 3.12, `pip install -e core[dev]`, `ruff check`,
     `pytest -m "not network and not paid and not gpu and not slow"`.
   - `.github/pull_request_template.md` — что меняется; затрагивает ли контракт ядра (да/нет); как проверено.
   - `.github/CODEOWNERS` — файлы контракта (`core/fastinfer/types.py`, `core/fastinfer/engines/base.py`,
     `core/fastinfer/__init__.py`, `core/templates/`, `docs/core_api.md`, `docs/result_example.json`) —
     на всех троих. GitHub-логины спросить у Игоря; до ответа оставить заглушки.
   - Лицензию не добавлять, пока команда не выберет (предложить MIT или Apache-2.0).
3. **`core/pyproject.toml`**: пакет `fastinfer`, версия 0.0.1, `requires-python = ">=3.10"`.
   Зависимости: `pyyaml`, `numpy`, `httpx`, `huggingface_hub`. Extras: `hf = ["torch", "transformers==5.19.0"]`,
   `api = ["openai"]`, `dev = ["pytest", "ruff"]`. Маркеры pytest: `network`, `paid`, `gpu`, `slow`.
4. **`core/fastinfer/types.py`** — dataclass-ы ровно как в `docs/core_api.md`, раздел 5. Без логики.
5. **`core/fastinfer/engines/__init__.py`** и **`engines/base.py`** — `Protocol Engine` из раздела 6.
6. **`core/fastinfer/__init__.py`** — функции из раздела 7.1: полные сигнатуры с типами, docstring по-русски
   (что делает, что принимает и возвращает, ссылка на раздел `core_api.md`), тело — `raise NotImplementedError`.
   Экспортировать типы из `types.py`.
7. **Остальные модули ядра не создавать** (`config.py`, `measure.py`, `engines/llamacpp.py` и т. д.) —
   их заведёт разработчик ядра так, как решит сам.
8. **`core/tests/test_contract.py`** — тесты, которые проходят уже сейчас:
   - пакет импортируется, все типы создаются со значениями по умолчанию;
   - `Engine` — это `Protocol` с методами из раздела 6;
   - каждая функция из раздела 7.1 существует и бросает `NotImplementedError`;
   - каждый `core/templates/*.yaml` читается `yaml.safe_load`, в нём есть обязательные поля раздела 3,
     а `mode`, `device` и `engine` — из допустимых значений;
   - `docs/result_example.json` читается и содержит `schema`, `job_id`, `commit`, `variants`, `comparison`.
9. **README в папках**:
   - `experiments/README.md` — как завести эксперимент: папка `ГГГГ-ММ-ДД-имя`, `README.md` (гипотеза,
     семейство, конфиг, результат, выводы, что дальше), `config.yaml`, `run.py` только при необходимости,
     в `results/` — только сводки; сырые данные — на HF.
   - `dispatcher/README.md`, `infra/README.md` — назначение и ссылка на `docs/infra.md`; кода пока нет.
   - `reports/README.md` — сюда выводы и еженедельные отчёты.

## Готово, когда

- `pip install -e "core[dev]"`, `ruff check` и `pytest` проходят локально; CI зелёный.
- В ядре нет реализации — только типы, интерфейс и функции с `NotImplementedError`.
- В репозитории нет секретов, весов и файлов больше 50 МБ.

## После первого коммита — напомнить Игорю

- Включить в настройках репозитория push protection и secret scanning.
- Включить защиту ветки `main`: только через PR, нужен одобряющий ревью.
- Закрепить репозиторий в профилях всех троих.
- Вписать в `CLAUDE.md` и `docs/project.md`, кто какую роль занимает, и логины в `CODEOWNERS`.

"""Интерфейс движка — docs/core_api.md, раздел 6.

Любой движок (llama.cpp, HF transformers, API Nebius, поддельный для тестов) умеет одно и то же,
поэтому код экспериментов не зависит от того, где считается модель.
Это общий контракт команды: менять — только через PR, который смотрят все трое.
"""

from types import TracebackType
from typing import Protocol

from ..types import Generation, GenParams, Prompt


class Engine(Protocol):
    name: str

    def start(self) -> None:
        """Загрузить модель или поднять сервер."""
        ...

    def stop(self) -> None:
        """Выгрузить модель или остановить сервер."""
        ...

    def generate(self, prompt: Prompt, gen: GenParams) -> list[Generation]:
        """Ответить на один промпт: по одной генерации на каждый ход (реплику) промпта."""
        ...

    def generate_many(self, prompts: list[Prompt], gen: GenParams,
                      parallel: int = 1) -> list[list[Generation]]:
        """Ответить на много промптов, до `parallel` одновременно; по списку генераций на промпт."""
        ...

    def tokenize(self, text: str) -> list[int]:
        """Перевести текст в номера токенов этой модели."""
        ...

    def info(self) -> dict:
        """Версия, аргументы запуска, устройство — всё, что нужно записать в env.json."""
        ...

    # `with load_model(...) as m:` — `start` при входе, `stop` при выходе (раздел 6).
    def __enter__(self) -> "Engine":
        """Вызвать `start` и вернуть сам движок."""
        ...

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        """Вызвать `stop`, даже если внутри `with` случилась ошибка."""
        ...

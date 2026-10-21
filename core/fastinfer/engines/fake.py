import random
import string
from subprocess import check_call
from time import sleep
from types import TracebackType
from typing import Protocol

from .. import Engine
from ..types import Generation, GenParams, Prompt


def stable_hash(s: str) -> int:
    h = 2166136261  # FNV-1a offset basis (32-bit)
    for c in s:
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF  # FNV prime, keep 32-bit
    return h

class FakeEngineV1(Engine):

    name: str = "Fake engine"


    def __init__(self, speed = 1, seed = 123):
        self.randomCl = random.Random(seed)
        self.random = self.randomCl.random
        self.generating_seed = int(self.random()*10000)

        self.model = "fake-v1"
        self.base_tok_s = 100.0*speed
        self.speed = speed
        self.flags = {}

        self.running: bool = False
        self.determined: bool = False;
        self.api_calls: int = 0

        self.token_lookup = []
        self.token_decode = []


    def _load(self):
        self.token_lookup = list(range(1000))
        self.randomCl.shuffle(self.token_lookup)
        self.token_decode = [""]*1000
        cnt = 0
        while cnt < 1000:
            nxt = ''.join(self.randomCl.choices(string.ascii_letters, k=int(3+(self.random()*4))))
            hsh = stable_hash(nxt)%1000
            lead = self.token_lookup[hsh]
            if len(self.token_decode[lead])==0:
                cnt+=1
                self.token_decode[lead] = nxt


    def start(self) -> None:
        """Загрузить модель или поднять сервер."""
        if self.running:
            return
        if not self.determined:
            self._load()
            self.determined = True
        sleep(0.25)
        self.running = True

    def stop(self) -> None:
        """Выгрузить модель или остановить сервер."""
        sleep(0.23)
        self.running = False
        print("Engine stopped")



    def _generate(self, prompt: Prompt, gen: GenParams, fake_speed: float) -> list[Generation]:
        """Ответить на один промпт: по одной генерации на каждый ход (реплику)."""
        self._check_valid()

        speed = self.speed*max(1.0, float(fake_speed))

        generations: list[Generation] = []

        for turn, user_text in enumerate(prompt.turns):
            n_prompt = len(self.tokenize(user_text))

            # Длина ответа: 8..32 токена, но не больше max_tokens.
            limit = max(0, gen.max_tokens)
            n_gen = min(limit, 8 + (sum(ord(ch) for ch in prompt.id) % 25))

            tokens: list[int] = []
            for _ in range(n_gen):
                tokens.append(self._next_token(tokens))

            text = " ".join(
                self.token_decode[token] or f"<{token}>"
                for token in tokens
            )

            # Синтетические тайминги: sleep выше — это грубая модель задержки.
            total_ms = (0.1 / speed) * 1000.0
            ttft_ms = total_ms * 0.2
            decode_ms = total_ms - ttft_ms
            tok_s = (n_gen / decode_ms) * 1000.0 if decode_ms > 0 else 0.0

            logprobs = None
            if gen.logprobs >= 2:
                logprobs = []
                for token in tokens:
                    margin = 0.05 + (token % 30) / 10.0
                    logprobs.append([
                        (token, 0.0),
                        ((token + 1) % 1000, -margin),
                    ])

            finish_reason = "length" if n_gen >= limit else "stop"
            if any(stop and stop in text for stop in gen.stop):
                finish_reason = "stop"

            generations.append(Generation(
                prompt_id=prompt.id,
                turn=turn,
                text=text,
                token_ids=tokens,
                n_prompt=n_prompt,
                n_gen=n_gen,
                n_thinking=0,
                n_answer=n_gen,
                ttft_ms=ttft_ms,
                decode_ms=decode_ms,
                total_ms=total_ms,
                tok_s=tok_s,
                draft_n=0,
                draft_accepted=0,
                logprobs=logprobs,
                finish_reason=finish_reason,
                cost_usd=0.0,
            ))

        return generations







    def _next_token(self, tokens: list[int]):
        sm = sum(tokens)+self.generating_seed
        return sm%1000






    def generate(self, prompt: Prompt, gen: GenParams) -> list[Generation]:
        """Ответить на один промпт: по одной генерации на каждый ход (реплику) промпта."""
        return self._generate(prompt, gen, 1)


    def _check_valid(self):
        if not self.running:
            raise RuntimeError("Fake Engine V1 called when stopped")


    def generate_many(self, prompts: list[Prompt], gen: GenParams,
                      parallel: int = 1) -> list[list[Generation]]:
        """Ответить на много промптов, до `parallel` одновременно; по списку генераций на промпт."""
        return list(self._generate(i, gen, min(8,parallel)) for i in prompts)

    def tokenize(self, text: str) -> list[int]:
        """Перевести текст в номера токенов этой модели."""
        for spl in ",.\n\t_(){}[]+/":
            text = " ".join(text.split(spl))
        return list(self.token_lookup[stable_hash(i)%1000] for i in text.split())

    def info(self) -> dict:
        """Версия, аргументы запуска, устройство — всё, что нужно записать в env.json."""
        return {
            "engine": "fake",
            "model": self.model,
            "argv": ["--fake", f"--tok-s={self.base_tok_s}"],
            "device": "cpu",
            "flags": dict(self.flags),
            "version": "0",
        }

    # `with load_model(...) as m:` — `start` при входе, `stop` при выходе (раздел 6).
    def __enter__(self) -> FakeEngineV1:
        """Вызвать `start` и вернуть сам движок."""
        self.start()
        print("Engine entering a scope")
        return self


    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        """Вызвать `stop`, даже если внутри `with` случилась ошибка."""
        self.stop()
        print("Engine exiting from a scope")




































































































































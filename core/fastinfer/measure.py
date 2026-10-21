




























from typing import Sequence

from core.fastinfer import Engine, Prompt, GenParams, Report, SpeedStats, Generation
import numpy as np


def warmup(engine: Engine, prompts: Sequence[Prompt], gen: GenParams, n: int = 2) -> None:
    """Два промпта вхолостую: загрузка ядер, аллокации, первые токены."""
    for prompt in list(prompts)[:n]:
        engine.generate(prompt, gen)







def _speed_stats(generations: list[Generation]) -> SpeedStats:
    speeds = np.array([g.tok_s for g in generations if g.tok_s > 0], dtype=float)
    if speeds.size == 0:
        raise ValueError("нет ни одной генерации с tok_s > 0")

    ttft = np.array([g.ttft_ms for g in generations], dtype=float)
    total = np.array([g.total_ms for g in generations], dtype=float) / 1000.0

    draft_n = sum(g.draft_n for g in generations)
    accepted = sum(g.draft_accepted for g in generations)
    acceptance = (accepted / draft_n) if draft_n > 0 else None

    # τ = сколько токенов приходится на один проход цели.
    n_gen = sum(g.n_gen for g in generations)
    passes = sum(g.n_gen - g.draft_accepted for g in generations)
    tokens_per_pass = (n_gen / passes) if accepted > 0 and passes > 0 else None

    p10, p50, p90 = np.percentile(speeds, [10, 50, 90])
    return SpeedStats(
        tok_s_median=float(p50),
        tok_s_p10=float(p10),
        tok_s_p90=float(p90),
        ttft_ms_median=float(np.median(ttft)),
        total_s_median=float(np.median(total)),
        acceptance=acceptance,
        tokens_per_pass=tokens_per_pass,
        frac_of_ceiling=None,          # заполнит run_experiment, когда знает модель и ПСП
    )


def _run_engine_measurement(engine: Engine, prompts: Sequence[Prompt], gen: GenParams,
                            repeats: int = 3):
    results = []
    for i in range(repeats):
        for prompt in prompts:
            results.append(engine.generate(prompt, gen))
    return results



def measure_speed(engine: Engine, prompts: Sequence[Prompt], gen: GenParams,
                  repeats: int = 3) -> Report:


    warmup(engine, prompts, gen)


    generations = _run_engine_measurement(engine, prompts, gen, repeats)

    return Report(
        variant=engine.name,
        mode="speed",
        generations=generations,
        speed=_speed_stats(generations),
        quality=None,
        engine_info=engine.info(),
    )




































































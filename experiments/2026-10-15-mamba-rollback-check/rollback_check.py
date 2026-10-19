"""
Проверка отката состояния Mamba для спекулятивного декодинга на крошечной NemotronH.

Что проверяем (на случайной модели из 6 слоёв: Mamba, MLP, внимание; работает на CPU за секунды):
  A. Подать 6 токенов одним проходом при уже заполненном кэше (как при проверке черновика)
     == тот же кусок в одном полном проходе по всей последовательности.
  B. Откат: снимок состояний -> 6 «чужих» токенов (отвергнутый черновик) -> восстановление
     -> правильные токены == полный проход. Для контроля то же самое без отката SSM-состояния.
  C. Путь «по одному токену» (обычное декодирование) против пути проверки (несколько токенов разом).

Как запустить:
  pip install torch "transformers==5.19.0"
  python rollback_check.py

Что ожидать (fp32): A и B ~1e-6, B без отката ~1 и больше, C ~1e-4..1e-3.
C — не баг: это два разных алгоритма одного расчёта (пошаговая рекуррентность против блочного scan).
Именно поэтому тест «жадный вывод совпадает токен в токен» на настоящей модели нужен с допуском.
"""
import torch
from transformers import NemotronHConfig, NemotronHForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # без предупреждений о медленном пути без mamba-ssm


def make_model(dtype: torch.dtype) -> NemotronHForCausalLM:
    cfg = NemotronHConfig(
        vocab_size=512, hidden_size=64, num_hidden_layers=6,
        hybrid_override_pattern="M-M*M-",  # M — Mamba-2, '-' — MLP, '*' — внимание (как в 9B, только меньше)
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128,
        mamba_num_heads=4, mamba_head_dim=16, n_groups=2, ssm_state_size=16, conv_kernel=4,
        chunk_size=8, max_position_embeddings=256, initializer_range=0.2,
    )
    torch.manual_seed(0)
    return NemotronHForCausalLM(cfg).eval().to(dtype)


def snapshot(past):
    """Снимок кэша: conv- и SSM-состояния слоёв Mamba + длина KV-кэша слоёв внимания."""
    snap = []
    for layer in past.layers:
        name = type(layer).__name__
        if name == "LinearAttentionLayer" and layer.is_recurrent_states_initialized.get(0):
            snap.append(("ssm", layer.conv_states[0].clone(), layer.recurrent_states[0].clone()))
        elif name == "DynamicLayer" and layer.is_initialized:
            snap.append(("attn", layer.keys.shape[-2]))
        else:
            snap.append(None)  # MLP-слои ничего не хранят
    return snap


def restore(past, snap) -> None:
    """Откат к снимку: вернуть состояния Mamba и обрезать KV-кэш до прежней длины."""
    for layer, s in zip(past.layers, snap):
        if s is None:
            continue
        if s[0] == "ssm":
            layer.conv_states[0] = s[1].clone()
            layer.recurrent_states[0].copy_(s[2])
        else:
            n = s[1]
            layer.keys = layer.keys[..., :n, :]
            layer.values = layer.values[..., :n, :]


@torch.no_grad()
def run(dtype: torch.dtype) -> None:
    m = make_model(dtype)
    torch.manual_seed(1)
    ids = torch.randint(0, 512, (1, 24))
    full = m(ids).logits.float()  # эталон: один проход по всей последовательности

    # A. prefill 10 токенов, затем 6 токенов одним проходом
    past = m(ids[:, :10], use_cache=True).past_key_values
    cont = m(ids[:, 10:16], past_key_values=past).logits.float()
    a = (cont - full[:, 10:16]).abs().max().item()

    # B. снимок -> отвергнутый черновик -> откат -> 2 принятых токена -> ещё 4
    past = m(ids[:, :10], use_cache=True).past_key_values
    snap = snapshot(past)
    junk = torch.randint(0, 512, (1, 6))
    m(junk, past_key_values=past)
    restore(past, snap)
    m(ids[:, 10:12], past_key_values=past)
    after = m(ids[:, 12:16], past_key_values=past).logits.float()
    b = (after - full[:, 12:16]).abs().max().item()

    # контроль: обрезаем только KV-кэш, а SSM-состояние оставляем «испорченным»
    past2 = m(ids[:, :10], use_cache=True).past_key_values
    m(junk, past_key_values=past2)
    for layer in past2.layers:
        if type(layer).__name__ == "DynamicLayer" and layer.is_initialized:
            layer.keys, layer.values = layer.keys[..., :10, :], layer.values[..., :10, :]
    m(ids[:, 10:12], past_key_values=past2)
    bad = m(ids[:, 12:16], past_key_values=past2).logits.float()
    b_bad = (bad - full[:, 12:16]).abs().max().item()

    # C. те же 6 токенов по одному (путь обычного декодирования)
    past3 = m(ids[:, :10], use_cache=True).past_key_values
    stepwise = torch.cat([m(ids[:, t:t + 1], past_key_values=past3).logits.float() for t in range(10, 16)], dim=1)
    c = (stepwise - cont).abs().max().item()
    agree = (stepwise.argmax(-1) == cont.argmax(-1)).float().mean().item()

    print(f"{str(dtype):15s} A: {a:.1e} | B: {b:.1e} (без отката SSM: {b_bad:.1e}) | "
          f"C: {c:.1e}, совпадение argmax {agree:.0%}")


if __name__ == "__main__":
    print("Максимальное расхождение логитов с эталоном (меньше — лучше)")
    for dt in (torch.float32, torch.float16):
        try:
            run(dt)
        except Exception as e:  # fp16 на CPU поддерживается не везде
            print(f"{str(dt):15s} пропущено: {type(e).__name__}: {str(e)[:120]}")

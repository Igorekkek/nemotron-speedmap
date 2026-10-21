from pathlib import Path

from fastinfer import ModelSpec
from fastinfer.shared import todo


class ModelException(Exception):

    def __init__(self, err: str):
        super().__init__(err)




AVAILABLE_MODELS: dict[str, ModelSpec] = {
    "nano-4b": ModelSpec(
        name="nano-4b",
        kind="dense",
        hf_repo=None,  # точное имя исходных весов ещё не подтверждено
        gguf_repo="nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        gguf_files={},  # имя файла под квант ищется при скачивании
        arch={
            "layers": 42,
            "mamba_layers": 21,
            "attention_layers": 4,
            "mlp_layers": 17,
            "d_model": 3136,
            "ffn": 12544,
            "vocab": 131072,
            "revision": None,
        },
        params_per_token=3.53e9,
        sampling={},  # рекомендации для 4B в docs/facts.md не зафиксированы
        reasoning_switch="template:enable_thinking",
        api_id=None,
        license="NVIDIA Open Model License",
    ),
    "nano-9b-v2": ModelSpec(
        name="nano-9b-v2",
        kind="dense",
        hf_repo="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        gguf_repo=None,  # своего GGUF нет: конвертация или выбранный готовый репозиторий
        gguf_files={},
        arch={
            "layers": 56,
            "mamba_layers": 27,
            "attention_layers": 4,
            "mlp_layers": 25,
            "d_model": 4480,
            "ffn": 15680,
            "heads": 40,
            "kv_groups": 8,
            "mamba_heads": 128,
            "mamba_head_dim": 80,
            "state": 128,
            "vocab": 131072,
            "revision": None,
        },
        params_per_token=8.30e9,
        sampling={
            "reasoning": {"temperature": 0.6, "top_p": 0.95},
            "no_reasoning": {"temperature": 0.0, "top_p": 1.0},
        },
        reasoning_switch="system:/think|/no_think",
        api_id=None,
        license="NVIDIA Open Model License",
    ),
    "nano-30b-a3b": ModelSpec(
        name="nano-30b-a3b",
        kind="moe",
        hf_repo="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        gguf_repo="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        gguf_files={},
        arch={
            "layers": 52,
            "mamba_layers": 23,
            "moe_layers": 23,
            "attention_layers": 6,
            "d_model": 2688,
            "experts": 128,
            "experts_per_token": 6,
            "expert_ffn": 1856,
            "heads": 32,
            "kv_groups": 2,
            "head_dim": 128,
            "mamba_heads": 64,
            "mamba_head_dim": 64,
            "state": 128,
            "vocab": 131072,
            "revision": None,
        },
        params_per_token=3.497e9,
        sampling={
            "reasoning": {"temperature": 1.0, "top_p": 1.0},
            "tools": {"temperature": 0.6, "top_p": 0.95},
            "no_reasoning": {"temperature": 0.0, "top_p": 1.0},
        },
        reasoning_switch="template:enable_thinking",
        api_id=None,  # имя уточняется через GET /v1/models
        license="NVIDIA Open Model License",
    ),
    "lightning-3.5": ModelSpec(
        name="lightning-3.5",
        kind="moe",
        hf_repo=None,  # репозиторий исходных весов ещё не подтверждён
        gguf_repo=None,  # нужен GGUF с MTP-слоями, пока не выбран
        gguf_files={},
        arch={
            "layers": 52,
            "mamba_layers": 23,
            "moe_layers": 23,
            "attention_layers": 6,
            "d_model": 2688,
            "experts": 128,
            "experts_per_token": 6,
            "expert_ffn": 1856,
            "vocab": 131072,
            "revision": None,
        },
        params_per_token=3.5e9,
        sampling={
            "reasoning": {"temperature": 1.0, "top_p": 1.0},
            "no_reasoning": {"temperature": 0.0, "top_p": 1.0},
        },
        reasoning_switch="template:enable_thinking",
        api_id=None,
        license="NVIDIA Open Model License",
    ),
    "super-3": ModelSpec(
        name="super-3",
        kind="moe",
        hf_repo=None,
        gguf_repo=None,
        gguf_files={},
        arch={
            "layers": 88,
            "d_model": 4096,
            "experts": 512,
            "experts_per_token": 22,
            "latent_moe": 1024,
            "mtp_layers": 2,
            "vocab": 131072,
            "revision": None,
        },
        params_per_token=12.7e9,
        sampling={},
        reasoning_switch="template:enable_thinking",
        api_id=None,  # только API Nebius; имя уточняется через GET /v1/models
        license="NVIDIA Open Model License",
    ),
    "super-3.5-vl": ModelSpec(
        name="super-3.5-vl",
        kind="moe",
        hf_repo=None,
        gguf_repo=None,
        gguf_files={},
        arch={
            "layers": 88,
            "d_model": 4096,
            "experts": 512,
            "experts_per_token": 22,
            "latent_moe": 1024,
            "mtp_layers": 2,
            "vocab": 131072,
            "revision": None,
        },
        params_per_token=12.7e9,
        sampling={},
        reasoning_switch="template:enable_thinking",
        api_id=None,
        license="NVIDIA Open Model License",
    ),
}



def resolve_model(name: str, quant: str | None = None) -> ModelSpec:
    if not name in AVAILABLE_MODELS:
        raise ModelException(f"unknown model: {name} - use any of \n"
                             f"{'\n'.join(AVAILABLE_MODELS.keys())}")
    result = AVAILABLE_MODELS[name]
    todo("check quant")
    return result

def parse_model_ref(ref: str) -> tuple[str, str | None]:
    # "nano-4b:Q4_K_M" -> ("nano-4b", "Q4_K_M")
    raise NotImplementedError("parse ref")

def list_models() -> list[str]:
    return list(AVAILABLE_MODELS.keys())

def quants_for(name: str) -> list[str]:
    raise NotImplementedError("list quants")



def fetch_model(spec: ModelSpec, quant: str, cache_dir: str | Path) -> Path:
    raise NotImplementedError("fetch_model")

def bytes_per_token(spec: ModelSpec, quant: str | None) -> float:
    raise NotImplementedError("bytes_per_token")
def sha256_file(path: Path) -> str:
    raise NotImplementedError("sha256_file")
def cache_root() -> Path:
    raise NotImplementedError("cache_root")




























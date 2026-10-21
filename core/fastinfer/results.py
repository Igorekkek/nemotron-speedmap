import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from fastinfer import ExperimentConfig, Generation
from fastinfer.shared import todo


from config_load import config_hash


class ExperimentWriter:

    SCHEMA = 1
    HEARTBEAT_FREQUENCY = 5.0

    def __init__(self, experiment: ExperimentConfig):
        self.started_at = None
        self.init_at: float = time.time()
        self.configs: ExperimentConfig = experiment
        self.last_heartbeat: float = self.init_at
        self.progress: float = -1

        self.name_prefix = f"run_{self.init_at}_"
        result_folder = f"results_{self.name_prefix}"
        log_file = f"log_{self.name_prefix}"

        self.pth_result_folder = Path("run_results")/Path(result_folder)
        self.pth_logs = self.pth_result_folder/Path("logs")
        self.pth_log_file = self.pth_logs/Path(log_file)


        self.pth_result_folder.mkdir(parents=True, exist_ok=True)
        self.pth_logs.mkdir(parents=True, exist_ok=True)
        self.pth_env_file = self.pth_result_folder/Path("env.json")
        self.pth_heartbeat_file = self.pth_result_folder/Path("heartbeat.json")
        self.pth_perprompt_file = self.pth_result_folder/Path("per_prompt.jsonl")


        self.write_env()


    def start(self):
        self.started_at = time.time()


        self.write_log("log started\n")
        self.heartbeat(0.0)

        self.write_env()





    def write_log(self, text):
        with open(self.pth_log_file, 'a') as log:
            log.write(text)





    def heartbeat(self, progress: float = -1):
        current_time = time.time()
        if current_time-self.last_heartbeat < self.HEARTBEAT_FREQUENCY:
            return
        self.last_heartbeat = current_time
        self._heartbeat(progress)
    def _heartbeat(self, progress: float):
        if progress<0:
            progress = self.progress
        self.progress = progress
        with open(self.pth_heartbeat_file, 'w') as heartbeat_file:
            heartbeat_file.write(f"Im alive (at {time.time()})\n"
                                 f"Progress: {progress}")

    def logs_path(self) -> Path:
        return self.pth_logs










    def write_perprompt(
            self,
            variant: str,
            repeat: int,
            generation: Generation,
            *,
            task: str,
            correct: bool | None = None,
            top2_margin: list[float] | None = None,
    ) -> None:
        """Дописать одну генерацию в per_prompt.jsonl (docs/core_api.md, раздел 4)."""
        record = {
            "variant": variant,
            "repeat": repeat,
            "prompt_id": generation.prompt_id,
            "task": task,
            "turn": generation.turn,
            "n_prompt": generation.n_prompt,
            "n_gen": generation.n_gen,
            "n_thinking": generation.n_thinking,
            "n_answer": generation.n_answer,
            "ttft_ms": generation.ttft_ms,
            "decode_ms": generation.decode_ms,
            "total_ms": generation.total_ms,
            "tok_s": generation.tok_s,
            "draft_n": generation.draft_n,
            "draft_accepted": generation.draft_accepted,
            "finish_reason": generation.finish_reason,
            "correct": correct,
            "cost_usd": generation.cost_usd,
            "text": generation.text,
        }

        # token_ids и top2_margin нужны только в первом повторе (раздел 4).
        if repeat == 0:
            record["token_ids"] = generation.token_ids
            record["top2_margin"] = top2_margin

        line = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,          # NaN/Infinity — невалидный JSON
            separators=(",", ":"),
        )
        with open(self.pth_perprompt_file, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")



    def write_env(
            self,
            *,
            job_id: str | None = None,
            commit: str | None = None,
            engine_info: dict | None = None,
            device: dict | None = None,
            models: dict | None = None,
            caps: dict | None = None,
    ) -> None:
        """Записать env.json: конфиг, версии, окружение по белому списку."""
        cfg = self.configs
        cfg_dict = asdict(cfg)


        allowlist = (
            "LLAMA_CPP_BIN",
            "FASTINFER_CACHE",
            "FASTINFER_JOB_ID",
            "FASTINFER_COMMIT",
            "FASTINFER_DEVICE",
            "FASTINFER_ENGINE",
            "FASTINFER_API_CAP_USD",
        )
        environment = {key: os.environ[key] for key in allowlist if key in os.environ}

        try:
            fastinfer_version = metadata.version("fastinfer")
        except metadata.PackageNotFoundError:
            fastinfer_version = "unknown"

        payload = {
            "schema": self.SCHEMA,
            "job_id": job_id or os.environ.get("FASTINFER_JOB_ID"),
            "commit": commit or os.environ.get("FASTINFER_COMMIT"),
            "name": cfg.name,
            "family": cfg.family,
            "mode": cfg.mode,
            "started_at": self.started_at or self.init_at,
            "config": cfg_dict,               # уже разрешённый: base → вариант, sweep развёрнут
            "config_hash": config_hash(cfg),
            "engine": engine_info or {},
            "device": device or {},
            "models": models or {},
            "caps": caps or {},
            "environment": environment,
            "versions": {
                "python": platform.python_version(),
                "fastinfer": fastinfer_version,
                "platform": platform.platform(),
            },
        }


        with open(self.pth_env_file, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, allow_nan=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

















































































































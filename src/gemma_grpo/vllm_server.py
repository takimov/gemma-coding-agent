from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import yaml

from .config import ExperimentConfig
from .nf4 import assert_nf4_checkpoint


BASE_MODEL_ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
ADAPTER_ALIAS = "main_lora"
VLLM_COMPAT_SITE = Path(__file__).with_name("_vllm_site")


def vllm_environment() -> dict[str, str]:
    """Return an environment that patches every spawned vLLM worker."""
    environment = os.environ.copy()
    environment.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    environment["VLLM_PLUGINS"] = ""
    environment["GEMMA_GRPO_VLLM_BNB_COMPAT"] = "1"
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(VLLM_COMPAT_SITE) + (
        os.pathsep + existing if existing else ""
    )
    return environment


def write_models_yaml(path: Path, *, port: int, adapter: Path | None) -> Path:
    api_base = f"http://127.0.0.1:{port}/v1"
    models: dict[str, dict[str, str]] = {
        BASE_MODEL_ALIAS: {
            "api_base": api_base,
            "api_key": "EMPTY",
            "path": f"openai/{BASE_MODEL_ALIAS}",
        }
    }
    if adapter is not None:
        models[ADAPTER_ALIAS] = {
            "api_base": api_base,
            "api_key": "EMPTY",
            "path": f"openai/{ADAPTER_ALIAS}",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"models": models}, sort_keys=False), encoding="utf-8")
    return path


class ManagedVllmServer:
    """Own a vLLM process so its VRAM is released before FSDP starts."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        model_path: Path,
        adapter: Path | None,
        work_dir: Path,
    ) -> None:
        self.config = config
        self.model_path = model_path.resolve()
        self.adapter = adapter.resolve() if adapter is not None else None
        self.work_dir = work_dir.resolve()
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: TextIO | None = None
        self.log_path = self.work_dir / "vllm.log"
        self.models_yaml = self.work_dir / "models.yaml"

    def _supported_flags(self) -> str:
        environment = vllm_environment()
        completed = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Could not inspect the installed vLLM server CLI:\n" + completed.stdout[-4000:]
            )
        return completed.stdout

    def _command(self) -> list[str]:
        flags = self._supported_flags()
        rollout = self.config.rollout
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.model_path),
            "--served-model-name",
            BASE_MODEL_ALIAS,
            "--port",
            str(rollout.vllm_port),
            "--tensor-parallel-size",
            str(rollout.vllm_tensor_parallel_size),
            "--pipeline-parallel-size",
            str(rollout.vllm_pipeline_parallel_size),
            "--gpu-memory-utilization",
            str(rollout.vllm_gpu_memory_utilization),
            "--max-model-len",
            str(self.config.grpo.max_seq_length),
        ]

        def add(flag: str, *values: str) -> None:
            if flag in flags:
                command.extend([flag, *values])

        add("--quantization", "bitsandbytes")
        add("--load-format", "bitsandbytes")
        add("--tool-call-parser", "gemma4")
        add("--reasoning-parser", "gemma4")
        add("--enable-auto-tool-choice")
        add("--disable-log-requests")
        if "--default-chat-template-kwargs" in flags:
            command.extend(
                ["--default-chat-template-kwargs", json.dumps({"enable_thinking": True})]
            )
        if self.adapter is not None:
            required = (
                "--enable-lora",
                "--lora-modules",
                "--max-lora-rank",
                "--max-loras",
            )
            missing = [flag for flag in required if flag not in flags]
            if missing:
                raise RuntimeError(
                    "Installed vLLM cannot serve the training adapter; missing CLI flags: "
                    + ", ".join(missing)
                )
            command.extend(
                [
                    "--enable-lora",
                    "--max-loras",
                    "1",
                    "--max-lora-rank",
                    str(self.config.lora.r),
                    "--lora-modules",
                    f"{ADAPTER_ALIAS}={self.adapter}",
                ]
            )
        return command

    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.config.rollout.vllm_port}/v1/models",
                timeout=2,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            ids = {str(item.get("id")) for item in payload.get("data", [])}
            required = ADAPTER_ALIAS if self.adapter is not None else BASE_MODEL_ALIAS
            return required in ids
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
            return False

    def _log_tail(self, limit: int = 8000) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return "(vLLM log unavailable)"

    def start(self) -> Path:
        assert_nf4_checkpoint(self.model_path, require_fsdp_storage=True)
        if self.adapter is not None:
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                if not (self.adapter / filename).is_file():
                    raise ValueError(f"Adapter {self.adapter} is missing {filename}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        write_models_yaml(
            self.models_yaml,
            port=self.config.rollout.vllm_port,
            adapter=self.adapter,
        )
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        environment = vllm_environment()
        command = self._command()
        print(f"[vllm] starting NF4 server; log={self.log_path}", flush=True)
        self.process = subprocess.Popen(
            command,
            text=True,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.config.rollout.vllm_startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                code = self.process.returncode
                self.stop()
                raise RuntimeError(
                    f"vLLM exited during startup with code {code}:\n{self._log_tail()}"
                )
            if self._health():
                print("[vllm] NF4 server ready", flush=True)
                return self.models_yaml
            time.sleep(2)
        self.stop()
        raise TimeoutError(
            "vLLM did not become ready within "
            f"{self.config.rollout.vllm_startup_timeout_seconds}s:\n{self._log_tail()}"
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if process is not None:
            print("[vllm] stopped; GPUs released for FSDP", flush=True)

    def __enter__(self) -> Path:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


@contextmanager
def rollout_models_yaml(
    *,
    config: ExperimentConfig,
    model_path: str | Path,
    adapter: Path | None,
    work_dir: Path,
    external_models_yaml: Path | None,
) -> Iterator[Path | None]:
    if external_models_yaml is not None:
        if config.rollout.managed_vllm:
            raise ValueError(
                "Choose one vLLM owner: set rollout.managed_vllm=false when passing --models-yaml"
            )
        yield external_models_yaml
        return
    if not config.rollout.managed_vllm:
        yield None
        return
    server = ManagedVllmServer(
        config=config,
        model_path=Path(model_path),
        adapter=adapter,
        work_dir=work_dir,
    )
    with server as models_yaml:
        yield models_yaml

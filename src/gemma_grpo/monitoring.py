from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


NVIDIA_QUERY = (
    "index,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "power.draw,temperature.gpu"
)


def _parse_nvidia_smi(output: str) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        try:
            numeric = [None if value in {"N/A", "[N/A]"} else float(value) for value in fields[1:]]
            rows.append(
                {
                    "index": int(fields[0]),
                    "utilization_percent": numeric[0],
                    "memory_utilization_percent": numeric[1],
                    "memory_used_mib": numeric[2],
                    "memory_total_mib": numeric[3],
                    "power_watts": numeric[4],
                    "temperature_c": numeric[5],
                }
            )
        except ValueError:
            continue
    return rows


def _process_rss_mib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _process_cpu_seconds() -> float | None:
    try:
        fields = Path("/proc/self/stat").read_text(encoding="utf-8").split()
        ticks = int(fields[13]) + int(fields[14])
        return ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _process_threads() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _torch_gpu_memory() -> dict[int, dict[str, float]]:
    """Read allocator stats without importing/initializing torch early."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {}
    rows = {}
    for index in range(torch.cuda.device_count()):
        rows[index] = {
            "torch_allocated_mib": torch.cuda.memory_allocated(index) / 1024**2,
            "torch_reserved_mib": torch.cuda.memory_reserved(index) / 1024**2,
            "torch_max_allocated_mib": torch.cuda.max_memory_allocated(index) / 1024**2,
        }
    return rows


def _system_memory() -> dict[str, float]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.split()[0])
    except (OSError, ValueError, IndexError):
        return {}
    return {
        "total_mib": values.get("MemTotal", 0) / 1024,
        "available_mib": values.get("MemAvailable", 0) / 1024,
    }


def _summarize_rows(rows: list[dict[str, Any]], *, phase_kind: bool) -> dict[str, Any]:
    phases: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        phase = str(row.get("phase", "unknown"))
        if phase_kind:
            phase = phase.split(":", 1)[0]
        bucket = phases.setdefault(
            phase,
            {
                "samples": 0,
                "duration_seconds": 0.0,
                "max_process_rss_mib": 0.0,
                "process_cpu_percent_sum": 0.0,
                "process_cpu_percent_samples": 0,
                "max_process_threads": 0,
                "min_system_available_mib": None,
                "gpus": {},
            },
        )
        bucket["samples"] += 1
        if index + 1 < len(rows):
            elapsed = float(row.get("elapsed_seconds", 0.0))
            next_elapsed = float(rows[index + 1].get("elapsed_seconds", elapsed))
            bucket["duration_seconds"] += max(0.0, next_elapsed - elapsed)
        rss = row.get("process_rss_mib")
        if isinstance(rss, (int, float)):
            bucket["max_process_rss_mib"] = max(bucket["max_process_rss_mib"], float(rss))
        cpu_percent = row.get("process_cpu_percent")
        if isinstance(cpu_percent, (int, float)):
            bucket["process_cpu_percent_sum"] += float(cpu_percent)
            bucket["process_cpu_percent_samples"] += 1
        threads = row.get("process_threads")
        if isinstance(threads, int):
            bucket["max_process_threads"] = max(bucket["max_process_threads"], threads)
        available = (row.get("system_memory") or {}).get("available_mib")
        if isinstance(available, (int, float)):
            previous = bucket["min_system_available_mib"]
            bucket["min_system_available_mib"] = (
                float(available) if previous is None else min(previous, float(available))
            )
        for gpu in row.get("gpus") or []:
            gpu_index = str(gpu["index"])
            gpu_bucket = bucket["gpus"].setdefault(
                gpu_index,
                {
                    "samples": 0,
                    "utilization_sum": 0.0,
                    "max_utilization_percent": 0.0,
                    "max_memory_used_mib": 0.0,
                    "memory_utilization_sum": 0.0,
                    "memory_utilization_samples": 0,
                    "power_sum_watts": 0.0,
                    "power_samples": 0,
                    "max_temperature_c": 0.0,
                    "max_torch_allocated_mib": 0.0,
                    "max_torch_reserved_mib": 0.0,
                },
            )
            utilization = float(gpu.get("utilization_percent") or 0.0)
            gpu_bucket["samples"] += 1
            gpu_bucket["utilization_sum"] += utilization
            gpu_bucket["max_utilization_percent"] = max(
                gpu_bucket["max_utilization_percent"], utilization
            )
            memory_used = gpu.get("memory_used_mib")
            if isinstance(memory_used, (int, float)):
                gpu_bucket["max_memory_used_mib"] = max(
                    gpu_bucket["max_memory_used_mib"], float(memory_used)
                )
            memory_utilization = gpu.get("memory_utilization_percent")
            if isinstance(memory_utilization, (int, float)):
                gpu_bucket["memory_utilization_sum"] += float(memory_utilization)
                gpu_bucket["memory_utilization_samples"] += 1
            power = gpu.get("power_watts")
            if isinstance(power, (int, float)):
                gpu_bucket["power_sum_watts"] += float(power)
                gpu_bucket["power_samples"] += 1
            temperature = gpu.get("temperature_c")
            if isinstance(temperature, (int, float)):
                gpu_bucket["max_temperature_c"] = max(
                    gpu_bucket["max_temperature_c"], float(temperature)
                )
            gpu_bucket["max_torch_allocated_mib"] = max(
                gpu_bucket["max_torch_allocated_mib"],
                float(gpu.get("torch_allocated_mib") or 0.0),
            )
            gpu_bucket["max_torch_reserved_mib"] = max(
                gpu_bucket["max_torch_reserved_mib"],
                float(gpu.get("torch_reserved_mib") or 0.0),
            )

    for bucket in phases.values():
        bucket["duration_seconds"] = round(bucket["duration_seconds"], 3)
        bucket["max_process_rss_mib"] = round(bucket["max_process_rss_mib"], 1)
        cpu_samples = bucket.pop("process_cpu_percent_samples")
        cpu_sum = bucket.pop("process_cpu_percent_sum")
        bucket["average_process_cpu_percent"] = round(
            cpu_sum / cpu_samples, 1
        ) if cpu_samples else None
        available = bucket["min_system_available_mib"]
        if available is not None:
            bucket["min_system_available_mib"] = round(available, 1)
        for gpu_bucket in bucket["gpus"].values():
            samples = gpu_bucket.pop("samples")
            gpu_bucket["average_utilization_percent"] = round(
                gpu_bucket.pop("utilization_sum") / samples, 1
            )
            memory_samples = gpu_bucket.pop("memory_utilization_samples")
            memory_sum = gpu_bucket.pop("memory_utilization_sum")
            gpu_bucket["average_memory_utilization_percent"] = (
                round(memory_sum / memory_samples, 1) if memory_samples else None
            )
            power_samples = gpu_bucket.pop("power_samples")
            power_sum = gpu_bucket.pop("power_sum_watts")
            gpu_bucket["average_power_watts"] = (
                round(power_sum / power_samples, 1) if power_samples else None
            )
            gpu_bucket["max_memory_used_mib"] = round(
                gpu_bucket["max_memory_used_mib"], 1
            )
    return phases


def summarize_telemetry(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Telemetry file is empty: {path}")
    return {
        "path": str(path),
        "samples": len(rows),
        "total_elapsed_seconds": round(float(rows[-1].get("elapsed_seconds", 0.0)), 3),
        "phase_kinds": _summarize_rows(rows, phase_kind=True),
        "phases": _summarize_rows(rows, phase_kind=False),
    }


class TrainingMonitor:
    """Sample process, system RAM, and every NVIDIA GPU into JSONL."""

    def __init__(
        self,
        output_path: Path,
        *,
        interval_seconds: float,
        console_seconds: float,
    ) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.console_seconds = console_seconds
        self._phase = "initializing"
        self._phase_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = time.monotonic()
        self._last_console = 0.0
        self._previous_cpu_seconds: float | None = None
        self._previous_cpu_wall: float | None = None

    def __enter__(self) -> "TrainingMonitor":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="grpo-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))
        with self._phase_lock:
            self._phase = "failed" if exc is not None else "complete"
        self._sample(force_console=True)

    @property
    def phase(self) -> str:
        with self._phase_lock:
            return self._phase

    def set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = phase
        self._sample(force_console=True)

    def _gpu_rows(self) -> list[dict[str, float | int | None]]:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={NVIDIA_QUERY}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                text=True,
                capture_output=True,
                timeout=4,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        rows = _parse_nvidia_smi(completed.stdout)
        torch_rows = _torch_gpu_memory()
        for row in rows:
            row.update(torch_rows.get(int(row["index"]), {}))
        return rows

    def _snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        cpu_seconds = _process_cpu_seconds()
        cpu_percent = None
        if (
            cpu_seconds is not None
            and self._previous_cpu_seconds is not None
            and self._previous_cpu_wall is not None
            and now > self._previous_cpu_wall
        ):
            cpu_percent = (
                (cpu_seconds - self._previous_cpu_seconds)
                / (now - self._previous_cpu_wall)
                * 100
            )
        self._previous_cpu_seconds = cpu_seconds
        self._previous_cpu_wall = now
        return {
            "unix_time": time.time(),
            "elapsed_seconds": now - self._started_at,
            "pid": os.getpid(),
            "phase": self.phase,
            "process_rss_mib": _process_rss_mib(),
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_percent": cpu_percent,
            "process_threads": _process_threads(),
            "system_load_average": list(os.getloadavg()),
            "system_memory": _system_memory(),
            "gpus": self._gpu_rows(),
        }

    def _sample(self, *, force_console: bool = False) -> None:
        with self._sample_lock:
            snapshot = self._snapshot()
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            now = time.monotonic()
            if force_console or now - self._last_console >= self.console_seconds:
                gpu_text = " ".join(
                    f"gpu{row['index']}={float(row.get('utilization_percent') or 0):.0f}%/"
                    f"{float(row.get('memory_used_mib') or 0) / 1024:.1f}GiB"
                    for row in snapshot["gpus"]
                )
                available = snapshot["system_memory"].get("available_mib")
                ram_text = f"ram_free={available / 1024:.1f}GiB" if available is not None else ""
                rss = snapshot["process_rss_mib"]
                rss_text = f"rss={rss / 1024:.1f}GiB" if rss is not None else ""
                cpu = snapshot["process_cpu_percent"]
                cpu_text = f"cpu={cpu:.0f}%" if cpu is not None else ""
                print(
                    f"[monitor] phase={snapshot['phase']} {gpu_text} "
                    f"{ram_text} {rss_text} {cpu_text}".strip(),
                    flush=True,
                )
                self._last_console = now

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

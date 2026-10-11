import json
import time

from gemma_grpo.monitoring import TrainingMonitor, _parse_nvidia_smi, summarize_telemetry


def test_parse_nvidia_smi_all_gpus():
    rows = _parse_nvidia_smi(
        "0, 97, 81, 20480, 22528, 65.5, 72\n"
        "1, 12, 20, 10240, 22528, 31.0, 54\n"
    )
    assert rows == [
        {
            "index": 0,
            "utilization_percent": 97.0,
            "memory_utilization_percent": 81.0,
            "memory_used_mib": 20480.0,
            "memory_total_mib": 22528.0,
            "power_watts": 65.5,
            "temperature_c": 72.0,
        },
        {
            "index": 1,
            "utilization_percent": 12.0,
            "memory_utilization_percent": 20.0,
            "memory_used_mib": 10240.0,
            "memory_total_mib": 22528.0,
            "power_watts": 31.0,
            "temperature_c": 54.0,
        },
    ]


def test_summarize_telemetry_by_phase(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    rows = [
        {
            "elapsed_seconds": 0,
            "phase": "load",
            "process_rss_mib": 100,
            "system_memory": {"available_mib": 1000},
            "gpus": [{"index": 0, "utilization_percent": 10, "memory_used_mib": 200, "power_watts": 30}],
        },
        {
            "elapsed_seconds": 5,
            "phase": "train",
            "process_rss_mib": 120,
            "system_memory": {"available_mib": 900},
            "gpus": [{"index": 0, "utilization_percent": 90, "memory_used_mib": 800, "power_watts": 70}],
        },
        {
            "elapsed_seconds": 10,
            "phase": "complete",
            "process_rss_mib": 110,
            "system_memory": {"available_mib": 950},
            "gpus": [{"index": 0, "utilization_percent": 0, "memory_used_mib": 100, "power_watts": 20}],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = summarize_telemetry(path)

    assert summary["total_elapsed_seconds"] == 10
    assert summary["phases"]["load"]["duration_seconds"] == 5
    assert summary["phases"]["train"]["gpus"]["0"]["average_utilization_percent"] == 90
    assert summary["phase_kinds"]["train"]["duration_seconds"] == 5


def test_training_monitor_persists_phase_transitions_without_nvidia_smi(tmp_path):
    path = tmp_path / "live.jsonl"
    with TrainingMonitor(path, interval_seconds=0.01, console_seconds=60) as monitor:
        monitor.set_phase("load:model")
        time.sleep(0.03)
        monitor.set_phase("train:seed_1")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    phases = {row["phase"] for row in rows}
    assert {"load:model", "train:seed_1", "complete"} <= phases

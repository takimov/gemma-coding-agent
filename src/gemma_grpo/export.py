from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


MAX_ARCHIVE_BYTES = 3 * 1024**3


def _attach_adapter(agent_path: Path) -> None:
    lines = agent_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("adapter:"):
            lines[index] = "adapter: main_lora"
            break
    else:
        model_index = next((index for index, line in enumerate(lines) if line.startswith("model:")), 1)
        lines.insert(model_index + 1, "adapter: main_lora")
    agent_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_submission(*, submission_dir: Path, adapter_dir: Path, output_zip: Path) -> Path:
    adapter_files = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in adapter_files if not (adapter_dir / name).is_file()]
    if missing:
        raise ValueError(f"Adapter {adapter_dir} is missing: {', '.join(missing)}")
    output_zip = output_zip.with_suffix(".zip")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gemma-grpo-export-") as temporary:
        staging = Path(temporary) / "submission"
        shutil.copytree(submission_dir, staging)
        adapter_target = staging / "adapters" / "main_lora"
        adapter_target.mkdir(parents=True)
        for name in adapter_files:
            shutil.copy2(adapter_dir / name, adapter_target / name)
        _attach_adapter(staging / "agent.yaml")
        archive_base = output_zip.with_suffix("")
        created = Path(shutil.make_archive(str(archive_base), "zip", root_dir=staging))
    if created.stat().st_size >= MAX_ARCHIVE_BYTES:
        created.unlink()
        raise ValueError("Submission archive exceeds the 3 GiB competition limit")
    return created


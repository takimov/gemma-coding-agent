# Kaggle 4×L4: FSDP + QLoRA + NF4 rollout

Notebook работает с выключенным Internet. Managed loop использует один
постоянный NF4 checkpoint и сам чередует vLLM TP=4 с FSDP `FULL_SHARD`=4.
Одновременно эти процессы не живут: перед training vLLM завершается и отдаёт
всю VRAM.

## Обновление private source dataset

На Mac, после commit:

```bash
python scripts/publish_kaggle_dataset.py
```

Publisher архивирует Git `HEAD`, пишет `commit.txt`, собирает зафиксированный
официальный `vllm-bnb-plugin` wheel и скачивает проверенный по SHA256 Linux-wheel
`bitsandbytes==0.48.1`. Plugin обязателен: в vLLM 0.19.1 NF4/BnB поддержка
вынесена из основного пакета. Dataset по умолчанию:
`timurakimoff/gemma-grpo-source`.

## Inputs

Подключите:

1. competition data `gemma-4-developer-agent`;
2. `google/gemma-4`, вариант `gemma-4-31b-it-qat-w4a16-ct`;
3. `metric/gemma-4-developer-agent-wheelhouse`;
4. `timurakimoff/gemma-grpo-source`;
5. со второй сессии — сохранённый NF4 checkpoint предыдущей версии notebook.

Accelerator: `GPU L4 ×4`. Internet: Off.

## Ячейка 1 — код и офлайн-зависимости

```python
from pathlib import Path
import glob
import json
import shutil
import subprocess
import sys
import zipfile

INPUT = Path("/kaggle/input")
PROJECT = Path("/kaggle/working/gemma-grpo")

wheel_dir = next(INPUT.rglob("swegemma-*.whl")).parent
plugin_wheel = next(INPUT.rglob("vllm_bnb_plugin-*.whl"))
asset_root = plugin_wheel.parent
bnb_wheel = next(asset_root.glob("bitsandbytes-0.48.1-*.whl"))
data_root = next(
    path.parent
    for path in INPUT.rglob("tasks.jsonl")
    if (path.parent / "snapshots").is_dir()
)
source_w4 = next(
    path.parent
    for path in INPUT.rglob("config.json")
    if "gemma-4-31b-it-qat-w4a16-ct" in str(path.parent)
)

shutil.rmtree(PROJECT, ignore_errors=True)
repo_roots = [
    path.parent
    for path in asset_root.rglob("pyproject.toml")
    if (path.parent / "src" / "gemma_grpo").is_dir()
]
archives = [
    path
    for path in asset_root.iterdir()
    if path.is_file()
    and path.suffix != ".whl"
    and zipfile.is_zipfile(path)
]
if repo_roots:
    source_root = min(repo_roots, key=lambda path: len(path.parts))
    shutil.copytree(source_root, PROJECT)
elif archives:
    PROJECT.mkdir(parents=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(PROJECT)
else:
    raise FileNotFoundError(f"gemma-grpo source is missing under {asset_root}")
assert (PROJECT / "pyproject.toml").is_file(), PROJECT

# Убираем конфликтующий cutlass bootstrap и устанавливаем полный официальный
# wheelhouse ровно как в competition image.
for pattern in (
    "/usr/local/lib/python*/dist-packages/*cutlass*.pth",
    "/usr/local/lib/python*/site-packages/*cutlass*.pth",
):
    for filename in glob.glob(pattern):
        Path(filename).unlink(missing_ok=True)

normalized = Path("/tmp/gemma-host-wheelhouse")
shutil.rmtree(normalized, ignore_errors=True)
normalized.mkdir()
for wheel in wheel_dir.glob("*.whl"):
    if "cutlass" in wheel.name.lower():
        continue
    name = (
        wheel.name.replace("cu128", "+cu128")
        if "cu128" in wheel.name and "+" not in wheel.name
        else wheel.name
    )
    (normalized / name).symlink_to(wheel)

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "--force-reinstall",
        *sorted(str(path) for path in normalized.glob("*.whl")),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "--force-reinstall", str(bnb_wheel),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "--force-reinstall", str(plugin_wheel),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-e", str(PROJECT),
        "--no-deps", "--no-build-isolation",
    ],
    check=True,
)

commit_file = asset_root / "commit.txt"
print("project:", PROJECT)
print("commit:", commit_file.read_text().strip())
print("source W4:", source_w4)
print("competition data:", data_root)
print("bitsandbytes:", bnb_wheel.name)
print("NF4 plugin:", plugin_wheel.name)
```

## Ячейка 2 — выбрать или один раз создать NF4 checkpoint

```python
import torch

assert torch.cuda.device_count() == 4, torch.cuda.device_count()
for index in range(4):
    props = torch.cuda.get_device_properties(index)
    print(index, props.name, round(props.total_memory / 2**30, 1), "GiB")

def is_nf4(path):
    try:
        config = json.loads((path / "config.json").read_text())
        quant = config.get("quantization_config") or {}
        return (
            str(quant.get("quant_method", "")).startswith("bitsandbytes")
            and quant.get("load_in_4bit") is True
            and quant.get("bnb_4bit_quant_type") == "nf4"
            and str(quant.get("bnb_4bit_quant_storage")) in {"bfloat16", "torch.bfloat16"}
        )
    except (OSError, ValueError, TypeError):
        return False

mounted_nf4 = [path.parent for path in INPUT.rglob("config.json") if is_nf4(path.parent)]
NF4_DIR = mounted_nf4[0] if mounted_nf4 else Path(
    "/kaggle/working/models/gemma-4-31b-nf4"
)

if not is_nf4(NF4_DIR):
    subprocess.run(
        [
            "gemma-grpo", "convert-nf4",
            "--source-model", str(source_w4),
            "--output-dir", str(NF4_DIR),
        ],
        check=True,
        cwd=PROJECT,
    )

print("persistent NF4 checkpoint:", NF4_DIR)
```

Это единственное место, где исходный W4 распаковывается и переквантуется.
Конвертация требует системную RAM и использует GPU 0 только как scratch. После
этого и vLLM, и FSDP читают уже сохранённый NF4. Сохраните `NF4_DIR` как output
первой notebook version и подключайте его как input в следующих сессиях.

## Ячейка 3 — preflight

```python
subprocess.run(
    [
        "gemma-grpo", "preflight",
        "--config", str(PROJECT / "configs/smoke.yaml"),
        "--tasks", str(data_root / "tasks.jsonl"),
        "--snapshots-dir", str(data_root / "snapshots"),
        "--submission-dir", str(PROJECT / "submission"),
        "--model-path", str(NF4_DIR),
    ],
    check=True,
    cwd=PROJECT,
)
```

Preflight должен подтвердить `vllm-bnb-plugin`, BNB ≥0.48.1, сохранённый NF4
с BF16 quant storage и четыре CUDA device. При любой красной проверке 31B не
запускайте.

## Ячейка 4 — обязательный smoke

```python
tasks = [
    json.loads(line)
    for line in (data_root / "tasks.jsonl").read_text().splitlines()
]
[(row["instance_id"], row["repo"]) for row in tasks[:30]]
```

```python
TASK_IDS = ["requests_7505"]
smoke_dir = Path("/kaggle/working/runs/smoke-fsdp-v1")

command = [
    "gemma-grpo", "loop",
    "--config", str(PROJECT / "configs/smoke.yaml"),
    "--tasks", str(data_root / "tasks.jsonl"),
    "--snapshots-dir", str(data_root / "snapshots"),
    "--submission-dir", str(PROJECT / "submission"),
    "--model-path", str(NF4_DIR),
    "--task-id", TASK_IDS[0],
    "--work-dir", str(smoke_dir),
    "--target-steps", "1",
    "--max-collections", "5",
]
if (smoke_dir / "loop_state.json").exists():
    command.append("--resume")
subprocess.run(command, check=True, cwd=PROJECT)
```

Ожидаемая последовательность лога:

```text
[vllm] starting NF4 server
[vllm] NF4 server ready
[loop] collection ...
[vllm] stopped; GPUs released for FSDP
[train] launching FSDP-QLoRA on 4 GPUs
```

В группе теперь четыре seed: один rollout попадает на один FSDP rank. В
training telemetry должны одновременно появляться все четыре GPU, а не прежняя
пила последовательного `device_map`.

```python
telemetry = max(
    Path("/kaggle/working/runs").rglob("*.telemetry*.jsonl"),
    key=lambda path: path.stat().st_mtime,
)
subprocess.run(
    ["gemma-grpo", "telemetry", "--input", str(telemetry)],
    check=True,
    cwd=PROJECT,
)
print("raw telemetry:", telemetry)
```

## Ячейка 5 — непрерывный run

```python
TASK_IDS = [
    "REPLACE_WITH_REAL_TASK_ID_1",
    "REPLACE_WITH_REAL_TASK_ID_2",
]
run_dir = Path("/kaggle/working/runs/grpo-fsdp-001")

command = [
    "gemma-grpo", "loop",
    "--config", str(PROJECT / "configs/grpo.yaml"),
    "--tasks", str(data_root / "tasks.jsonl"),
    "--snapshots-dir", str(data_root / "snapshots"),
    "--submission-dir", str(PROJECT / "submission"),
    "--model-path", str(NF4_DIR),
    "--work-dir", str(run_dir),
    "--target-steps", "100",
    "--max-collections", "200",
]
for task_id in TASK_IDS:
    command.extend(["--task-id", task_id])
if (run_dir / "loop_state.json").exists():
    command.append("--resume")

subprocess.run(command, check=True, cwd=PROJECT)
```

Не передавайте `--models-yaml`: managed loop сам владеет сервером. Kaggle run
ограничен 12 часами, поэтому разумно сначала ставить cumulative target 10,
сохранять `/kaggle/working/runs/grpo-fsdp-001`, затем продолжать с `--resume`.

## Ячейка 6 — export

```python
state = json.loads((run_dir / "loop_state.json").read_text())
adapter = state["current_adapter"]
assert adapter, state

submission_zip = Path("/kaggle/working/submission.zip")
subprocess.run(
    [
        "gemma-grpo", "export",
        "--submission-dir", str(PROJECT / "submission"),
        "--adapter", adapter,
        "--output", str(submission_zip),
    ],
    check=True,
    cwd=PROJECT,
)
print(submission_zip, round(submission_zip.stat().st_size / 2**20, 1), "MiB")
```

Перед submission отдельно проверьте adapter с официальным W4 base: competition
evaluator использует W4, тогда как on-policy train/rollout теперь согласованы на
NF4. Совместимость имён LoRA-модулей сохраняется, но logits двух base formats не
bit-exact.

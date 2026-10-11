# Kaggle 4×L4: native W4 rollout + FSDP-QLoRA NF4 training

Notebook работает с выключенным Internet. Managed loop чередует быстрый native
QAT W4A16 vLLM (`TP=4`) с FSDP `FULL_SHARD`=4 на постоянном NF4 checkpoint.
Оба checkpoint'а существуют заранее; между фазами переносится только adapter.
Перед training vLLM завершается и отдаёт всю VRAM.

## Обновление private source dataset

На Mac, после commit:

```bash
python scripts/publish_kaggle_dataset.py
```

Publisher архивирует Git `HEAD`, пишет `commit.txt` и скачивает проверенный по
SHA256 Linux-wheel `bitsandbytes==0.48.1`. vLLM 0.19.1 использует собственный
встроенный BNB loader. Dataset по умолчанию:
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
bnb_wheel = next(INPUT.rglob("bitsandbytes-0.48.1-*.whl"))
asset_root = bnb_wheel.parent
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
```

## Ячейка 2 — выбрать или один раз создать NF4 checkpoint

```python
import torch

assert torch.cuda.device_count() == 4, torch.cuda.device_count()
for index in range(4):
    props = torch.cuda.get_device_properties(index)
    print(index, props.name, round(props.total_memory / 2**30, 1), "GiB")

def is_nf4(path, storage):
    try:
        config = json.loads((path / "config.json").read_text())
        quant = config.get("quantization_config") or {}
        return (
            str(quant.get("quant_method", "")).startswith("bitsandbytes")
            and quant.get("load_in_4bit") is True
            and quant.get("bnb_4bit_quant_type") == "nf4"
            and str(quant.get("bnb_4bit_quant_storage")) in storage
        )
    except (OSError, ValueError, TypeError):
        return False

mounted_nf4 = [
    path.parent
    for path in INPUT.rglob("config.json")
    if is_nf4(path.parent, {"bfloat16", "torch.bfloat16"})
]
NF4_DIR = mounted_nf4[0] if mounted_nf4 else Path(
    "/kaggle/working/models/gemma-4-31b-nf4"
)

if not is_nf4(NF4_DIR, {"bfloat16", "torch.bfloat16"}):
    subprocess.run(
        [
            "gemma-grpo", "convert-nf4",
            "--source-model", str(source_w4),
            "--output-dir", str(NF4_DIR),
        ],
        check=True,
        cwd=PROJECT,
    )

print("persistent training NF4 checkpoint:", NF4_DIR)
```

Только первый шаг распаковывает W4 и квантует training-веса. Для rollout vLLM
читает исходный `source_w4` прямо из Kaggle Input; NF4 используется FSDP.
Сохраните NF4 directory как notebook output для следующих сессий.

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
        "--rollout-model-path", str(source_w4),
    ],
    check=True,
    cwd=PROJECT,
)
```

Preflight должен подтвердить BNB ≥0.48.1, training NF4 checkpoint, rollout
`compressed-tensors` W4 checkpoint и четыре CUDA device.
Managed server также включает `--language-model-only`: SWEGemma coding tasks не
содержат изображений, а vision tower этого checkpoint не нужен для rollout.
Rollout context остаётся 32768 независимо от `grpo.max_seq_length`: smoke
фильтрует обучающие траектории по 8192. На каждом tool-turn SWEGemma резервирует
до 8192 новых токенов, оставляя примерно 24k контекста под накопленную историю.
Не возвращайте `max_output_tokens: 16384`: при истории длиннее 16k vLLM отклонит
запрос, потому что prompt и зарезервированный completion вместе превысят 32768.
Вывод managed vLLM одновременно показывается в stdout notebook и сохраняется в
`<run>/vllm/collection_NNNN/vllm.log`.
Native W4 использует `TP=4`, CUDA graphs и `max_num_seqs=4`. Четыре seed'а
запускаются в независимых SWEGemma subprocess sandbox одновременно, а один
vLLM непрерывно батчит их запросы. Один evaluator-процесс всё равно получает
`--concurrency 1`, потому что в нём ровно одна задача; `rollout.concurrency`
управляет параллелизмом seed'ов. Это тот же быстрый inference-класс, на котором
были собраны первые smoke rollout'ы с разными rewards. NF4/BitsAndBytes runtime
shim в этой фазе намеренно не загружается.
После появления reward variance loop параллельно делает короткий scoring pass
по action tokens и сохраняет W4 raw-model `behavior_logprobs`; в stdout виден
прогресс `[rollout] scoring W4 behavior logprobs N/4`.
Managed server передаёт Gemma 4 reasoning boundaries (`<|channel>` и
`<channel|>`) через `--reasoning-config`, поэтому запросы SWEGemma с
`thinking_token_budget` принимаются vLLM 0.19.1.

## Быстрый offline SFT

SFT полезен, когда online GRPO слишком дорог: rollout'ы собираются один раз,
после чего все optimizer steps выполняются подряд без vLLM. Нельзя обучаться на
неуспешных траекториях автоматически — команда принимает только
`resolved=true` либо вручную проверенные строки с `sft_approved=true`.

Сначала можно использовать уже накопленные JSONL и проверить объём learn:

```python
rollout_paths = sorted(Path("/kaggle/working/runs").rglob("rollouts.jsonl"))
rows = [
    json.loads(line)
    for path in rollout_paths
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
approved = [
    row for row in rows
    if (row.get("resolved") is True and row.get("scorable", True) is not False)
    or row.get("sft_approved") is True
]
print("trajectory files:", len(rollout_paths))
print("rows:", len(rows))
print("approved for SFT:", len(approved))
print("tasks:", len({row.get("task_id") for row in approved}))
assert approved, "Пока нет ни одной успешной/одобренной траектории для SFT"
```

Если данных мало, соберите несколько задач одним запуском vLLM. Флаг
`--skip-behavior-logprobs` убирает W4 scoring pass, который нужен GRPO, но не
SFT:

```python
SFT_TASK_IDS = [
    # Добавьте разные instance_id из tasks.jsonl.
    "requests_7505",
]
sft_data_dir = Path("/kaggle/working/runs/sft-data-v1")
command = [
    "gemma-grpo", "collect",
    "--config", str(PROJECT / "configs/grpo.yaml"),
    "--tasks", str(data_root / "tasks.jsonl"),
    "--snapshots-dir", str(data_root / "snapshots"),
    "--submission-dir", str(PROJECT / "submission"),
    "--model-path", str(source_w4),
    "--results-root", str(sft_data_dir / "results"),
    "--output", str(sft_data_dir / "rollouts.jsonl"),
    "--iteration", "0",
    "--skip-behavior-logprobs",
]
for task_id in SFT_TASK_IDS:
    command.extend(["--task-id", task_id])
subprocess.run(command, check=True, cwd=PROJECT)
```

Обучение по всем накопленным JSONL (global batch равен четырём GPU):

```python
rollout_paths = sorted(Path("/kaggle/working/runs").rglob("rollouts.jsonl"))
SFT_OUTPUT = Path("/kaggle/working/runs/sft-adapter-v1")
command = [
    "gemma-grpo", "sft",
    # Первый запуск: sft-smoke.yaml (ровно один step). После него: sft.yaml.
    "--config", str(PROJECT / "configs/sft-smoke.yaml"),
    "--trajectories", *map(str, rollout_paths),
    "--output-dir", str(SFT_OUTPUT),
    "--model-path", str(NF4_DIR),
]
subprocess.run(command, check=True, cwd=PROJECT)
print(json.loads((SFT_OUTPUT / "training_metrics.json").read_text()))
```

Production-конфиг `configs/sft.yaml` использует две эпохи,
`max_seq_length=8192`, LoRA rank 16 и не более двух уникальных траекторий с
одной задачи. Готовый `configs/sft-smoke.yaml` делает ровно один step с той же
LoRA-архитектурой. После успешного smoke можно либо начать чистый production
run, либо передать smoke-adapter через `--resume-adapter` и продолжить в новом
output directory с `configs/sft.yaml`.

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
    "--rollout-model-path", str(source_w4),
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
[vllm] starting compressed-tensors server
[vllm] compressed-tensors server ready
[loop] collection ...
[vllm] stopped; GPUs released for FSDP
[train] launching FSDP-QLoRA on 4 GPUs
```

В группе теперь четыре seed: один rollout попадает на один FSDP rank. В
training telemetry должны одновременно появляться все четыре GPU, а не прежняя
пила последовательного `device_map`.
В `training_metrics.json` проверьте `importance_clip_fraction`: это доля action
tokens, для которых различие W4 behavior и NF4 policy вышло за GRPO clip range.

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
    "--rollout-model-path", str(source_w4),
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

Перед submission отдельно проверьте adapter с официальным W4 base. Rollout и
competition evaluator используют W4, training — NF4; сохранённые W4 raw-model
log-probabilities корректируют это различие в clipped ratio, но logits двух base
formats всё равно не bit-exact.

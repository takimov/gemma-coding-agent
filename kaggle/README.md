# Запуск GRPO на Kaggle L4 x4

## Быстрая синхронизация из GitHub

Во время отладки удобнее держать Internet включённым и получать исправления из
публичного GitHub-репозитория. Эта ячейка создаёт отдельный checkout при первом
запуске, а затем делает только fast-forward pull, поэтому результаты обучения в
`/kaggle/working/runs` не затрагиваются:

```python
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/takimov/gemma-coding-agent.git"
BRANCH = "main"
PROJECT = Path("/kaggle/working/gemma-grpo-live")

if (PROJECT / ".git").is_dir():
    subprocess.run(
        ["git", "-C", str(PROJECT), "pull", "--ff-only", "origin", BRANCH],
        check=True,
    )
else:
    if PROJECT.exists():
        raise RuntimeError(f"Refusing to overwrite non-git directory: {PROJECT}")
    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--branch", BRANCH,
            REPO_URL, str(PROJECT),
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

commit = subprocess.check_output(
    ["git", "-C", str(PROJECT), "rev-parse", "--short", "HEAD"],
    text=True,
).strip()
print("project:", PROJECT)
print("commit:", commit)
```

Команды `gemma-grpo`, запускаемые после этого через `subprocess`, читают новую
версию сразу. Уже импортированные в kernel модули `gemma_grpo` остаются в
`sys.modules`; после обновления для прямых Python-импортов перезапустите kernel.
Не выполняйте `git pull` посреди активного `collect` или `train`.

Для notebook version, которая обязана работать с выключенным Internet,
используйте ZIP/Kaggle Dataset из следующего раздела.

После каждого проверенного commit приватный source dataset можно обновить с
локального Mac официальным Kaggle CLI:

```bash
python scripts/publish_kaggle_dataset.py
```

Publisher отказывается брать dirty worktree, архивирует именно Git `HEAD` и
пишет полный commit hash в `commit.txt`. Dataset по умолчанию:
`timurakimoff/gemma-grpo-source`.

## Inputs ноутбука

Создайте notebook из страницы соревнования и подключите четыре input:

1. competition data `gemma-4-developer-agent`;
2. модель `google/gemma-4` с вариантом `gemma-4-31b-it-qat-w4a16-ct`;
3. официальный dataset `metric/gemma-4-developer-agent-wheelhouse`;
4. private dataset `timurakimoff/gemma-grpo-source` с файлом
   `gemma-grpo-kaggle.zip` из этого проекта.

В Settings выберите `GPU L4 x4` и оставьте Internet выключенным.

## Ячейка 1: найти mounts, распаковать код и установить пакеты

```python
from pathlib import Path
import glob
import shutil
import subprocess
import sys
import zipfile

INPUT = Path("/kaggle/input")
PROJECT = Path("/kaggle/working/gemma-grpo")

wheel_dir = next(INPUT.rglob("swegemma-*.whl")).parent
data_root = next(
    path.parent
    for path in INPUT.rglob("tasks.jsonl")
    if (path.parent / "snapshots").is_dir()
)
model_dir = next(
    path.parent
    for path in INPUT.rglob("config.json")
    if "gemma-4-31b-it-qat-w4a16-ct" in str(path.parent)
)

shutil.rmtree(PROJECT, ignore_errors=True)

source_archives = list(INPUT.rglob("gemma-grpo-kaggle.zip"))
source_directories = [
    path for path in INPUT.rglob("gemma-grpo-kaggle") if path.is_dir()
]
if source_archives:
    PROJECT.mkdir(parents=True)
    with zipfile.ZipFile(source_archives[0]) as archive:
        archive.extractall(PROJECT)
elif source_directories:
    # Kaggle commonly expands an uploaded ZIP and mounts it as a directory
    # whose name no longer has the .zip suffix.
    shutil.copytree(source_directories[0], PROJECT)
else:
    source_root = next(
        path.parent
        for path in INPUT.rglob("pyproject.toml")
        if "gemma-swegemma-grpo" in path.read_text(encoding="utf-8")
    )
    shutil.copytree(source_root, PROJECT)

assert (PROJECT / "pyproject.toml").is_file(), (
    f"Source was not materialized correctly under {PROJECT}"
)

# The official host wheelhouse is a self-contained, mutually compatible bundle.
# Install all of it instead of naming vLLM dependencies one by one.  The
# cutlass wheel is intentionally excluded, matching the competition example:
# its .pth bootstrap conflicts with Kaggle's preinstalled CUDA environment.
for pattern in (
    "/usr/local/lib/python*/dist-packages/*cutlass*.pth",
    "/usr/local/lib/python*/site-packages/*cutlass*.pth",
):
    for path in glob.glob(pattern):
        Path(path).unlink(missing_ok=True)

normalized_wheels = Path("/tmp/gemma-host-wheelhouse")
shutil.rmtree(normalized_wheels, ignore_errors=True)
normalized_wheels.mkdir(parents=True)
for wheel in wheel_dir.glob("*.whl"):
    if "cutlass" in wheel.name.lower():
        continue
    # A few supplied CUDA wheels need a PEP 440 local-version separator.
    name = (
        wheel.name.replace("cu128", "+cu128")
        if "cu128" in wheel.name and "+" not in wheel.name
        else wheel.name
    )
    (normalized_wheels / name).symlink_to(wheel)

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "--force-reinstall",
        *sorted(str(path) for path in normalized_wheels.glob("*.whl")),
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

print("project:", PROJECT)
print("competition data:", data_root)
print("model:", model_dir)
print("wheelhouse:", wheel_dir)
```

В комплектных `smoke.yaml` и `grpo.yaml` выбран официальный `subprocess`
sandbox SWEGemma. Kaggle не предоставляет Docker daemon: с `sandbox: docker`
rollout падает до старта model server, поэтому utilization GPU остаётся нулевой.

Не фиксируйте номер версии wheelhouse в коде: организаторы обновляют его.
Используется именно версия dataset, подключённая к notebook.

## Ячейка 2: проверить четыре GPU и окружение

```python
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(index, props.name, round(props.total_memory / 2**30, 1), "GiB")
assert torch.cuda.device_count() == 4, "Notebook запущен не на L4 x4"
```

```python
subprocess.run(
    [
        "gemma-grpo", "preflight",
        "--config", str(PROJECT / "configs/grpo.yaml"),
        "--tasks", str(data_root / "tasks.jsonl"),
        "--snapshots-dir", str(data_root / "snapshots"),
        "--submission-dir", str(PROJECT / "submission"),
        "--model-path", str(model_dir),
    ],
    check=True,
)
```

## Ячейка 3: выбрать задачи

Сначала посмотрите доступные идентификаторы и выберите несколько задач, а не
все 129 сразу.

```python
import json

tasks = [json.loads(line) for line in (data_root / "tasks.jsonl").read_text().splitlines()]
[(row["instance_id"], row["repo"]) for row in tasks[:30]]
```

```python
TASK_IDS = [
    "REPLACE_WITH_REAL_TASK_ID_1",
    "REPLACE_WITH_REAL_TASK_ID_2",
]
```

## Ячейка 4: обязательный smoke test

```python
smoke_dir = Path("/kaggle/working/runs/smoke-v2")
command = [
    "gemma-grpo", "loop",
    "--config", str(PROJECT / "configs/smoke.yaml"),
    "--tasks", str(data_root / "tasks.jsonl"),
    "--snapshots-dir", str(data_root / "snapshots"),
    "--submission-dir", str(PROJECT / "submission"),
    "--model-path", str(model_dir),
    "--task-id", TASK_IDS[0],
    "--work-dir", str(smoke_dir),
    "--target-steps", "1",
    "--max-collections", "5",
]
if (smoke_dir / "loop_state.json").exists():
    command.append("--resume")
subprocess.run(command, check=True, cwd=PROJECT)
```

Каждый `train` внутри loop печатает строки `[monitor]` и сохраняет полный JSONL
рядом с adapter directory. После smoke найдите последний файл и получите сводку:

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

Если train упадёт, telemetry всё равно заканчивается фазой `failed`. Для текущего
model-parallel режима нормально видеть последовательные пики GPU 0→1→2→3;
ненормально — CPU/disk offload, VRAM вплотную к 22 GiB или отсутствие активности
у GPU, которому device map назначил decoder layers.

## Ячейка 5: непрерывный запуск до 100 optimizer steps

Эта ячейка допустима только если подключённая версия harness сама завершает
свой model server после каждого `swegemma eval`. Она **не** управляет сервером,
который вы вручную создали через `adk_submission` и указали в local
`models.yaml`; оставленный процесс на `127.0.0.1:8000` вызовет OOM при backward.
Для такого режима сначала нужен managed server lifecycle (см. основной аудит).

```python
run_dir = Path("/kaggle/working/runs/grpo-001")

# Если к новой версии подключён output предыдущего run, восстановить checkpoint.
previous_checkpoints = list(INPUT.rglob("grpo-001-checkpoint.zip"))
if previous_checkpoints and not (run_dir / "loop_state.json").exists():
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(previous_checkpoints[0]) as archive:
        archive.extractall(run_dir.parent)

command = [
    "gemma-grpo", "loop",
    "--config", str(PROJECT / "configs/grpo.yaml"),
    "--tasks", str(data_root / "tasks.jsonl"),
    "--snapshots-dir", str(data_root / "snapshots"),
    "--submission-dir", str(PROJECT / "submission"),
    "--model-path", str(model_dir),
    "--work-dir", str(run_dir),
    "--target-steps", "100",
    "--max-collections", "100",
]
for task_id in TASK_IDS:
    command.extend(["--task-id", task_id])

# Если есть предварительно обученный SFT adapter:
# command.extend(["--initial-adapter", "/kaggle/input/MY-SFT-DATASET/adapter"])

if (run_dir / "loop_state.json").exists():
    command.append("--resume")

subprocess.run(command, check=True, cwd=PROJECT)
```

Один процесс сам чередует rollout и backward. Вкладку можно закрыть, если
ноутбук запущен через Save Version / Save & Run All. Один Kaggle GPU run имеет
ограничение 12 часов, поэтому 100 steps могут не поместиться в одну сессию.
После каждого train-update состояние лежит в `loop_state.json`, а текущий LoRA
adapter — в `runs/grpo-001/adapters/`.

Для первого фонового `Save & Run All` разумнее поставить `--target-steps 10`.
В следующей версии подключите output предыдущей версии и увеличьте cumulative
target до `20`, затем до `30` и так далее. Это не ручное чередование
collect/train: ограничение связано только с 12-часовой границей Kaggle.

Сохраните resume-checkpoint как notebook output:

```python
checkpoint_zip = shutil.make_archive(
    "/kaggle/working/grpo-001-checkpoint",
    "zip",
    root_dir=run_dir.parent,
    base_dir=run_dir.name,
)
print(checkpoint_zip)
```

## Ячейка 6: собрать submission.zip

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
print(submission_zip, submission_zip.stat().st_size / 2**20, "MiB")
```

Перед отправкой сохраните notebook version вместе с output. Для продолжения в
новой 12-часовой сессии подключите output предыдущей версии как input, скопируйте
папку `runs/grpo-001` обратно в `/kaggle/working/runs/grpo-001` и запустите ту же
команду: `--resume` добавится автоматически.

# SWEGemma GRPO: FSDP + QLoRA

Экспериментальный on-policy GRPO-loop для соревнования Gemma 4 Developer Agent.
Rollout и обучение используют один постоянный bitsandbytes NF4 checkpoint;
31B-модель больше не распаковывается и не переквантуется перед каждым update.

## Архитектура

Один процесс автоматически чередует две взаимоисключающие фазы на 4×L4:

1. managed vLLM загружает NF4 checkpoint tensor-parallel на четыре GPU и
   собирает четыре SWEGemma rollout'а;
2. vLLM полностью останавливается и освобождает VRAM;
3. четыре train-процесса загружают тот же NF4 checkpoint, FSDP `FULL_SHARD`
   распределяет параметры/градиенты/optimizer state, а QLoRA обновляет adapter;
4. adapter сохраняется, vLLM запускается с ним, и цикл повторяется.

NF4 относится к большим Linear-весам. Frozen embedding/lm_head и служебные
тензоры остаются BF16 — это стандартное и необходимое исключение FSDP-QLoRA.
`bnb_4bit_quant_storage=bfloat16` не распаковывает веса: четыре 4-битных значения
упакованы в один BF16 storage element, который FSDP умеет шардировать.

Смена фаз всё ещё перечитывает веса с диска: vLLM и backward 31B не помещаются
одновременно на тех же L4. Но преобразования W4 → BF16 → NF4 в цикле больше нет.

## Команды

- `gemma-grpo convert-nf4` — одноразово создаёт постоянный NF4 checkpoint;
- `gemma-grpo preflight` — проверяет checkpoint, 4 GPU, версии и vLLM BNB plugin;
- `gemma-grpo collect` — запускает managed NF4 vLLM и собирает rollout-группу;
- `gemma-grpo train` — один FSDP-QLoRA GRPO update;
- `gemma-grpo loop` — непрерывно чередует collect/train до целевого числа steps;
- `gemma-grpo telemetry` — агрегирует GPU/CPU/RAM telemetry;
- `gemma-grpo export` — собирает submission ZIP с LoRA adapter.

## Одноразовая конвертация

Competition checkpoint — `compressed-tensors` QAT W4. Его нужно преобразовать
ровно один раз, до smoke test:

```bash
gemma-grpo convert-nf4 \
  --source-model /path/to/gemma-4-31b-it-qat-w4a16-ct \
  --output-dir /path/to/gemma-4-31b-nf4
```

Конвертер распаковывает source на CPU, последовательно квантует Linear-слои на
одной scratch GPU и сохраняет обычный pre-quantized bitsandbytes checkpoint.
Повторный вызов не перезаписывает существующую папку. Для следующих Kaggle
сессий NF4 directory надо сохранить как notebook output/dataset; иначе дорогую
одноразовую операцию придётся повторить после очистки `/kaggle/working`.

## Smoke и непрерывное обучение

```bash
gemma-grpo preflight \
  --config configs/smoke.yaml \
  --tasks /path/to/tasks.jsonl \
  --snapshots-dir /path/to/snapshots \
  --submission-dir submission \
  --model-path /path/to/gemma-4-31b-nf4

gemma-grpo loop \
  --config configs/smoke.yaml \
  --tasks /path/to/tasks.jsonl \
  --snapshots-dir /path/to/snapshots \
  --submission-dir submission \
  --model-path /path/to/gemma-4-31b-nf4 \
  --task-id ACTUAL_TASK_ID \
  --work-dir runs/smoke-fsdp \
  --target-steps 1 \
  --max-collections 5
```

После успешного smoke замените config на `configs/grpo.yaml`, добавьте несколько
task IDs и увеличьте `--target-steps`. `--resume` продолжает атомарно сохранённый
`loop_state.json`; rollout с одинаковыми rewards автоматически пересэмплируется.
Внешний `--models-yaml` допустим только при `rollout.managed_vllm: false`.

## Что считается в GRPO

Для каждой задачи собираются четыре траектории. Reward официального SWEGemma
нормализуется внутри группы. Старые policy log-probabilities и frozen-reference
log-probabilities фиксируются до первого optimizer step. Loss считается только
на assistant/action токенах, с clipped ratio и KL penalty.

Если ATIF trace содержит token IDs каждого assistant call, используются они;
иначе JSONL явно помечается `chat_template_reconstruction`. Разное число
assistant calls между FSDP ranks выравнивается короткими zero-loss forwards —
без этого порядок FSDP collectives мог бы зависнуть.

Полный `action_tokens × vocabulary` tensor не создаётся. Hidden states сначала
выбираются по action positions, а tied `lm_head + log_softmax` считается
chunk'ами. Frozen tied embedding/lm_head реплицирован на каждом rank: это около
2.8 GiB BF16 на GPU, зато checkpoint recompute не обращается к уже reshared
FSDP-параметру.

## Установка

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --no-build-isolation -e '.[dev,train]'
PYTHONPATH=src pytest -q
```

В Kaggle vLLM 0.19+ требует отдельный `vllm-bnb-plugin`. Publisher кладёт
зафиксированный официальный plugin wheel в private source dataset, потому что
Internet с L4×4 выключен. Без этого wheel preflight завершится до загрузки 31B.

Подробные готовые notebook cells находятся в
[`kaggle/README.md`](kaggle/README.md), а допущения и остаточные риски — в
[`docs/training-audit.md`](docs/training-audit.md).

## Важные ограничения

- Hardware smoke FSDP+QLoRA на конкретном Kaggle image обязателен: локальные
  CPU tests не эмулируют 4×L4, NCCL и BNB kernels.
- Final evaluator использует официальный W4 base. NF4 rollout/train теперь
  согласованы между собой, но перед submission нужен отдельный regression eval
  adapter'а на W4 base.
- Реконструкция chat template не гарантирует bit-exact совпадение токенов vLLM;
  для длинного run предпочтительны полные ATIF token IDs.
- 32k context не считается подтверждённым, пока telemetry smoke не покажет
  запас VRAM. Начинайте с `configs/smoke.yaml` (8192).

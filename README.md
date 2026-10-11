# SWEGemma GRPO: FSDP + QLoRA

Экспериментальный GRPO-loop для соревнования Gemma 4 Developer Agent. Быстрый
rollout использует официальный QAT W4A16 `compressed-tensors` checkpoint, а
FSDP-QLoRA обучает один постоянный bitsandbytes NF4 checkpoint. Полные веса не
конвертируются между итерациями: между фазами переносится только LoRA adapter.

## Архитектура

Один процесс автоматически чередует две взаимоисключающие фазы на 4×L4:

1. managed vLLM читает официальный QAT W4A16 checkpoint и собирает четыре
   SWEGemma rollout'а через tensor parallelism на четырёх GPU;
2. vLLM полностью останавливается и освобождает VRAM;
3. четыре train-процесса загружают BF16-storage NF4 checkpoint, FSDP `FULL_SHARD`
   распределяет параметры/градиенты/optimizer state, а QLoRA обновляет adapter;
4. adapter сохраняется, vLLM запускается с ним, и цикл повторяется.

LoRA ограничена языковыми `q/k/v/o` и MLP projections; `vision_tower` явно
исключён, потому что rollout-сервер работает в `--language-model-only`.

NF4 относится к большим Linear-весам. Frozen embedding/lm_head и служебные
тензоры остаются BF16 — это стандартное и необходимое исключение FSDP-QLoRA.
`bnb_4bit_quant_storage=bfloat16` не распаковывает веса: четыре 4-битных значения
упакованы в один BF16 storage element, который FSDP умеет шардировать.

Смена фаз всё ещё перечитывает свои checkpoint'ы: vLLM и backward 31B не
помещаются одновременно на тех же L4. Но преобразования W4 → BF16 → NF4 в
цикле нет. W4 смонтирован из Kaggle Input, NF4 создаётся один раз.

## Команды

- `gemma-grpo convert-nf4` — одноразово создаёт постоянный NF4 checkpoint;
- `gemma-grpo preflight` — проверяет checkpoint, runtime shim, 4 GPU и версии;
- `gemma-grpo collect` — запускает managed native W4 vLLM и собирает rollout-группу;
- `gemma-grpo train` — один FSDP-QLoRA GRPO update;
- `gemma-grpo sft` — offline FSDP-QLoRA по успешным или вручную одобренным траекториям;
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
  --model-path /path/to/gemma-4-31b-nf4 \
  --rollout-model-path /path/to/gemma-4-31b-it-qat-w4a16-ct

gemma-grpo loop \
  --config configs/smoke.yaml \
  --tasks /path/to/tasks.jsonl \
  --snapshots-dir /path/to/snapshots \
  --submission-dir submission \
  --model-path /path/to/gemma-4-31b-nf4 \
  --rollout-model-path /path/to/gemma-4-31b-it-qat-w4a16-ct \
  --task-id ACTUAL_TASK_ID \
  --work-dir runs/smoke-fsdp \
  --target-steps 1 \
  --max-collections 5
```

После успешного smoke замените config на `configs/grpo.yaml`, добавьте несколько
task IDs и увеличьте `--target-steps`. `--resume` продолжает атомарно сохранённый
`loop_state.json`; rollout с одинаковыми rewards автоматически пересэмплируется.
Внешний `--models-yaml` допустим только при `rollout.managed_vllm: false`.

## Offline SFT вместо online GRPO

SFT использует тот же NF4 checkpoint, FSDP и LoRA, но не требует групп reward,
behavior/reference log-probabilities или повторного запуска vLLM между шагами.
В learn автоматически входят только строки с `resolved=true` и успешным
scoring. Вручную проверенную teacher-траекторию можно добавить с
`sft_approved=true`; обычные `tests_failed` и `empty_patch` игнорируются.

При сборе данных для SFT отключите ненужный GRPO scoring pass:

```bash
gemma-grpo collect \
  --config configs/grpo.yaml \
  --tasks /path/to/tasks.jsonl \
  --snapshots-dir /path/to/snapshots \
  --submission-dir submission \
  --model-path /path/to/gemma-4-31b-it-qat-w4a16-ct \
  --output runs/sft-data/rollouts.jsonl \
  --results-root runs/sft-data/results \
  --iteration 0 \
  --skip-behavior-logprobs \
  --task-id TASK_A --task-id TASK_B --task-id TASK_C
```

Затем обучите adapter сразу на нескольких ранее собранных JSONL:

```bash
gemma-grpo sft \
  --config configs/sft.yaml \
  --trajectories runs/*/rollouts.jsonl \
  --output-dir runs/sft-adapter-v1 \
  --model-path /path/to/gemma-4-31b-nf4
```

Loss — token-mean negative log-likelihood только по assistant/action tokens.
User/system/tool-output токены остаются контекстом, но не являются targets.
Точные дубли удаляются, а `sft.max_examples_per_task` не позволяет одной лёгкой
задаче занять learn всеми четырьмя seed'ами. Последний неполный global batch не
отбрасывается: свободные FSDP ranks выполняют синхронизированный zero-loss
forward.

## Qwen3-Coder-Next W4 teacher

Для rejection-sampling SFT можно собирать траектории более сильным offline
teacher: `Qwen3-Coder-Next` (80B total, 3B active) в заранее подготовленном
`compressed-tensors` W4A16 checkpoint. Используйте
`configs/teacher-qwen.yaml` и `submission-qwen`; финальный submission всё равно
остаётся Gemma 4.

```bash
gemma-grpo collect \
  --config configs/teacher-qwen.yaml \
  --tasks /path/to/tasks.jsonl \
  --snapshots-dir /path/to/snapshots \
  --submission-dir submission-qwen \
  --model-path /path/to/Qwen3-Coder-Next.w4a16 \
  --results-root runs/qwen-teacher/results \
  --output runs/qwen-teacher/rollouts.jsonl \
  --iteration 0 \
  --skip-behavior-logprobs \
  --task-id TASK_A
```

Teacher запускается с TP=4, контекстом 32k и parser `qwen3_coder`. Его ATIF
token IDs намеренно не сохраняются: vocab Qwen и Gemma различается, поэтому
SFT восстанавливает сообщения и токенизирует teacher actions Gemma-токенизатором.
В обучение по-прежнему автоматически допускаются только `resolved=true`
траектории. Это verified teacher SFT; полноценный token-level OPD между двумя
разными vocabulary без отдельного выравнивания logits не заявляется.

## Что считается в GRPO

Для каждой задачи собираются четыре траектории. Reward официального SWEGemma
нормализуется внутри группы. Для native W4 rollout vLLM отдельно записывает
W4 raw-model log-probabilities наблюдённых action tokens через
`prompt_logprobs=0`. Trainer использует их в clipped importance ratio, а frozen-reference
log-probabilities считает на NF4 base. Loss считается только на
assistant/action токенах.

`training_metrics.json` содержит `mean_abs_logprob_shift`,
`mean_importance_ratio` и `importance_clip_fraction`. На первом update они
показывают реальное расхождение W4 behavior policy и NF4 training policy.

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

Competition wheelhouse содержит слишком старый `bitsandbytes`, поэтому
publisher кладёт проверенный Linux-wheel `bitsandbytes==0.48.1` в private source
dataset. vLLM 0.19.1 уже имеет встроенный BNB loader; более новый внешний
`vllm-bnb-plugin` с ним несовместим и managed server явно отключает его через
`VLLM_PLUGINS=""`.

Подробные готовые notebook cells находятся в
[`kaggle/README.md`](kaggle/README.md), а допущения и остаточные риски — в
[`docs/training-audit.md`](docs/training-audit.md).

## Важные ограничения

- Hardware smoke FSDP+QLoRA на конкретном Kaggle image обязателен: локальные
  CPU tests не эмулируют 4×L4, NCCL и BNB kernels.
- Rollout W4A16 и training NF4 не bit-exact: update слегка off-policy. Держите
  `epochs` небольшим, обновляйте rollout после каждого adapter update и перед
  длинным run сравните качество/log-prob обеих политик на нескольких traces.
- Final evaluator использует тот же официальный W4 base, что и rollout, поэтому
  adapter обязательно проверяется там уже во время smoke после первого update.
- Реконструкция chat template не гарантирует bit-exact совпадение токенов vLLM;
  для длинного run предпочтительны полные ATIF token IDs.
- 32k context не считается подтверждённым, пока telemetry smoke не покажет
  запас VRAM. Начинайте с `configs/smoke.yaml` (8192).

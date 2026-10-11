# SWEGemma GRPO loop

Практический каркас для обучения LoRA-адаптера агента в соревновании
**Gemma 4 Developer Agent**. Он использует официальный `swegemma` как источник
rollout'ов и финального reward, а затем делает одну on-policy GRPO-итерацию по
assistant/action-токенам.

## Что именно реализовано

Цикл устроен так:

1. Для каждой задачи `swegemma eval` запускается четыре раза с разными seed.
2. Из `task_results.jsonl`, patch и ATIF trace собирается единый JSONL.
   В него записывается SHA-256 fingerprint policy-адаптера.
   Если ATIF содержит `prompt_token_ids`/`completion_token_ids`, trainer использует
   точные per-call токены; иначе явно помечает fallback на реконструкцию chat template.
3. Reward берётся из результата официальной pytest-верификации:
   `1` за resolved, `0` за валидный, но не прошедший patch, `-0.25` за пустой
   patch. Небольшой штраф за tool calls применяется только к успешным решениям.
4. Reward нормализуется внутри группы одной задачи.
5. Перед обновлением пересчитываются log-probabilities текущей policy и frozen
   base model для всего batch до первого optimizer step. Затем выполняются
   clipped GRPO updates LoRA-параметров относительно этой snapshot-policy.
6. Обновлённый адаптер снова кладётся в submission; harness сам подключает его
   к своему vLLM, и цикл повторяется на свежих rollout'ах.

Это намеренно не маскируется под готовый `TRL GRPOTrainer`: официальный harness
сам владеет двухконтейнерным agent loop, а его trace не содержит пригодных для
PPO старых token log-probabilities. Здесь они честно пересчитываются и
замораживаются для всего rollout-batch до первого шага оптимизатора. Несколько
`grpo.epochs` внутри одного запуска допустимы; после завершения запуска нужны
свежие rollout'ы.

## Структура

- `submission/` — минимальный declarative agent для `adk-submission`.
- `configs/grpo.yaml` — LoRA r=16/alpha=32, group size 4, beta 0.01 и лимиты.
- `gemma-grpo collect` — канонические rollout'ы через `swegemma`.
- `gemma-grpo train` — artifact-based GRPO update.
- `gemma-grpo export` — ZIP с агентом и адаптером для Kaggle.
- `gemma-grpo preflight` — проверки wheelhouse, данных, rank и chat template.
- `gemma-grpo loop` — непрерывно чередует свежие rollout'ы и GRPO до заданного
  числа optimizer steps.

## 1. Установка

Локально для разработки:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --no-build-isolation -e '.[dev,train]'
pytest
```

В Kaggle интернет выключен, поэтому harness ставится из wheelhouse, который
лежит в competition dataset. Точный mount меняется между версиями; сначала
найдите его:

```bash
find /kaggle/input -type f \( -name 'swegemma*.whl' -o -name 'adk_submission*.whl' \)
```

Затем:

```bash
pip install --no-index --find-links=/kaggle/input/COMPETITION_DATA/wheels \
  swegemma adk-submission adk-eval-core
pip install --no-build-isolation -e '.[train]'
```

Для воспроизводимости сохраняйте вывод `pip freeze` рядом с экспериментом.
В опубликованном starter-наборе встречались `swegemma 0.2.7`,
`adk-submission 0.2.11` и `adk-eval-core 0.1.0`, но preflight проверяет
фактически установленную версию, а не полагается на эти номера.

## 2. Preflight

```bash
gemma-grpo preflight \
  --config configs/grpo.yaml \
  --tasks /kaggle/input/COMPETITION_DATA/tasks.jsonl \
  --snapshots-dir /kaggle/input/COMPETITION_DATA/snapshots \
  --model-path /kaggle/input/ACTUAL_MODEL_DIRECTORY
```

Особенно важна проверка `chat template prefix preservation`. Если она падает,
нельзя продолжать обучение: маска assistant-токенов будет неверной. Путь модели
в `configs/grpo.yaml` — пример; перед запуском подставьте реальный Kaggle mount.

## 3. Сбор rollout'ов

В каноническом пути не нужно запускать vLLM вручную: `swegemma` и
`adk-submission` сами поднимают tensor-parallel server, валидируют submission и
регистрируют найденные LoRA-адаптеры. В competition notebook реестр модели уже
настроен. В другом окружении передайте поддерживаемый harness-файл через
`--models-yaml /path/to/models.yaml`.

Сначала обязательно smoke test на одной задаче:

```bash
gemma-grpo collect \
  --config configs/smoke.yaml \
  --tasks /kaggle/input/COMPETITION_DATA/tasks.jsonl \
  --snapshots-dir /kaggle/input/COMPETITION_DATA/snapshots \
  --task-id ACTUAL_TASK_ID \
  --iteration 0 \
  --output rollouts/smoke.jsonl
```

После проверки — полноценная группа:

```bash
gemma-grpo collect \
  --config configs/grpo.yaml \
  --tasks /kaggle/input/COMPETITION_DATA/tasks.jsonl \
  --snapshots-dir /kaggle/input/COMPETITION_DATA/snapshots \
  --task-id ACTUAL_TASK_ID \
  --iteration 0 \
  --output rollouts/iter_0000.jsonl
```

Каждый seed получает отдельный result directory. В Kaggle-конфигах используется
официальный `subprocess` sandbox SWEGemma, потому что Kaggle не предоставляет
Docker daemon. Ошибка evaluation worker/sandbox записывается как
`scorable=false` и не превращается в ложный negative reward. Для локального
запуска с Docker поменяйте `rollout.sandbox` обратно на `docker`.

### Непрерывный режим без ручного collect/train

Для обычного запуска отдельные команды ниже не нужны. Один launcher сам
переключает GPU между harness/vLLM и backward, обновляет adapter, меняет seeds и
собирает свежие rollout'ы до указанного числа optimizer steps:

```bash
gemma-grpo loop \
  --config configs/grpo.yaml \
  --tasks /kaggle/input/COMPETITION_DATA/tasks.jsonl \
  --snapshots-dir /kaggle/input/COMPETITION_DATA/snapshots \
  --model-path /kaggle/input/ACTUAL_MODEL_DIRECTORY \
  --task-id TASK_ID_1 \
  --task-id TASK_ID_2 \
  --task-id TASK_ID_3 \
  --work-dir runs/grpo-001 \
  --target-steps 100
```

Для старта с SFT LoRA добавьте:

```bash
  --initial-adapter outputs/sft_adapter
```

Состояние атомарно сохраняется в `runs/grpo-001/loop_state.json`. После
прерывания тот же запуск продолжается с `--resume`. Если группа имеет одинаковые
reward и нулевой advantage, launcher не делает фиктивный optimizer step, а
автоматически собирает следующую группу с другим набором seed.

## 4. GRPO update

Дождитесь завершения `collect`, чтобы запущенный harness model server освободил
GPU. Server и backward 31B-модели обычно не помещаются одновременно на тех же
четырёх L4.

```bash
gemma-grpo train \
  --config configs/grpo.yaml \
  --model-path /kaggle/input/ACTUAL_MODEL_DIRECTORY \
  --rollouts rollouts/iter_0000.jsonl \
  --output-dir outputs/adapter_iter_0001
```

Trainer обновляет только LoRA-параметры. Для длинных последовательностей он
требует модельный `forward(..., logits_to_keep=...)`, чтобы не материализовать
огромный tensor логитов для всей 32k-трассы. Если установленная версия
Transformers этого не умеет, запуск останавливается до OOM; для smoke test можно
уменьшить `max_seq_length` до 4096–8192.

В Kaggle-конфигах используется `model.load_mode: bnb4`. Обычный BF16 checkpoint
при этом загружается стандартным QLoRA-путём. Competition checkpoint в формате
`compressed-tensors` сначала распаковывается в системной RAM, а затем каждый
Linear по отдельности переквантовывается в bitsandbytes NF4 на scratch GPU и
возвращается на CPU до финального распределения по четырём L4. Поэтому на GPU
base остаётся 4-битной, но это не bit-exact исходный QAT W4: между rollout и
trainer base появляется небольшая ошибка повторного квантования.

Группы с одинаковым reward имеют нулевой advantage и пропускаются. Для реального
сигнала стоит выбирать задачи, где текущая policy иногда решает задачу, а иногда
нет. Если resolved rate строго 0%, сначала полезнее SFT/trajectory distillation
на reference patches, чем увеличивать число бессигнальных GRPO-rollout'ов.

## 5. Следующая итерация

Соберите свежую policy-группу, передав адаптер. Collector временно помещает его
в `submission/adapters/main_lora`, а harness сам регистрирует его в vLLM:

```bash
gemma-grpo collect \
  --config configs/grpo.yaml \
  --tasks /kaggle/input/COMPETITION_DATA/tasks.jsonl \
  --snapshots-dir /kaggle/input/COMPETITION_DATA/snapshots \
  --task-id ACTUAL_TASK_ID \
  --iteration 1 \
  --adapter outputs/adapter_iter_0001 \
  --output rollouts/iter_0001.jsonl
```

Не запускайте `train` повторно на том же JSONL с уже изменённым adapter. Несколько
внутренних `grpo.epochs` используют одну зафиксированную behavior-policy; после
завершения команды нужны свежие rollout'ы.

Следующий update обязан стартовать с policy, которая породила rollout. Команда
проверит fingerprint и остановится при несовпадении:

```bash
gemma-grpo train \
  --config configs/grpo.yaml \
  --model-path /kaggle/input/ACTUAL_MODEL_DIRECTORY \
  --resume-adapter outputs/adapter_iter_0001 \
  --rollouts rollouts/iter_0001.jsonl \
  --output-dir outputs/adapter_iter_0002
```

## 6. Экспорт submission

```bash
gemma-grpo export \
  --adapter outputs/adapter_iter_0001 \
  --output dist/submission.zip
```

Экспорт добавляет `adapter: main_lora` в `agent.yaml`, кладёт PEFT adapter в
`adapters/main_lora/` и проверяет лимит архива 3 GiB. Перед отправкой обязательно
прогоните этот ZIP тем же `swegemma eval`, который установлен в competition
notebook.

## Ограничения, которые нельзя игнорировать

- Это экспериментальный trainer для competition harness, не официальный код
  Google/Kaggle и не гарантия совместимости любого wheel/model revision.
- Полный GRPO на 31B — тяжёлый workload. Начинайте с 1–3 задач, 2 rollout'ов и
  4k context; только после успешного backward переходите к группе 4 и 32k.
- `device_map="auto"` + LoRA подходит для первичного QLoRA-style эксперимента,
  но для устойчивого многодневного обучения лучше FSDP/DeepSpeed и отдельный
  rollout cluster.
- Для честной оценки держите закрытый validation split и не выбирайте checkpoint
  по public leaderboard.

Полезные первичные материалы: [официальный Getting Started notebook](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent/input),
[competition page](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview),
[TRL GRPO documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer).

Пошаговый перенос проекта в competition notebook с L4 x4 описан в
`kaggle/README.md`.


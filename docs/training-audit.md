# Аудит FSDP-QLoRA GRPO path

Дата: 2026-10-11. Цель: Gemma 4 31B, Kaggle 4×NVIDIA L4 22 GiB,
rollout и training из одного сохранённого NF4 checkpoint.

## Новый lifecycle

```text
saved BF16-storage NF4 base
    ├─ streaming lossless uint8 views → vLLM built-in BNB, PP=4
    │                                  → SWEGemma rollouts → stop
    └─ 4 train ranks, FSDP FULL_SHARD + QLoRA → save LoRA adapter
                                                    │
                                                    └─ next rollout
```

Оригинальный competition W4 checkpoint участвует только в одноразовой
`convert-nf4`. Каждый следующий collect/train перечитывает один NF4 checkpoint
с диска. vLLM bitcast'ит только загружаемые packed tensors в RAM; W4 не
распаковывается, повторной квантизации и второй 17 GiB копии на диске нет.

## Что исправлено

| Приоритет | Проблема | Решение |
|---|---|---|
| P0 | Последовательный `device_map` нагружал GPU по очереди | Четыре FSDP rank, `FULL_SHARD`, одна trajectory/rank |
| P0 | Каждый update делал W4 → BF16 → NF4 | Один постоянный pre-quantized NF4 checkpoint |
| P0 | Rollout был W4, training NF4 | Managed vLLM читает те же NF4-байты через streaming uint8 view без requantization |
| P0 | Внешний BNB plugin новее vLLM 0.19.1 | Используется встроенный loader 0.19.1, внешний plugin отключён через `VLLM_PLUGINS=""` |
| P0 | vLLM 0.19 не поддерживает pre-quantized BNB при TP>1 | Rollout использует PP=4 и сохраняет quantized tensors целыми |
| P0 | vLLM создавал dense vision layers, но checkpoint хранил их в NF4 | Для текстовых SWEGemma задач включён `--language-model-only`; vision tower не создаётся |
| P0 | Полный FP32 `action_tokens × 262144` tensor давал OOM 6.42 GiB | Selective hidden states и chunked lm_head/log-softmax |
| P0 | Разное число ATIF segments на ranks нарушает порядок FSDP collectives | Tiny zero-loss padding forwards участвуют и в backward graph |
| P1 | Checkpointed lm_head мог обратиться к уже reshared weight | Frozen tied embedding/lm_head реплицирован BF16 и исключён из FSDP |
| P1 | Trainable parameter внутри ignored projection не синхронизировался бы | Явная проверка и отказ до training |
| P1 | Partial ATIF IDs обучали только часть trajectory | Exact path только при token IDs всех assistant calls |
| P1 | Advantage полной группы сохранялся после удаления длинной trajectory | При oversize пропускается вся группа |
| P1 | NaN/Inf попадал в optimizer | Finite checks objective и gradient norm |

## Память и форматы

`bnb_4bit_quant_storage=torch.bfloat16` не означает BF16-веса. BitsAndBytes
упаковывает четыре 4-битных значения в один BF16 storage element; float storage
нужен потому, что FSDP не шардирует integer parameters. Compute dtype тоже BF16.

Не квантуются:

- tied token embedding/lm_head — frozen BF16, около 2.8 GiB на каждом GPU;
- layer norms и малые служебные tensors;
- LoRA A/B — trainable floating parameters;
- optimizer state только для LoRA.

Поэтому «всё NF4» здесь означает все крупные подходящие Linear base weights,
как в стандартном QLoRA, а не буквальный NF4 для каждого tensor модели.

## GRPO и distributed correctness

Группа из четырёх rollout'ов делится по ranks как `prepared[rank::4]`. Каждый
rank считает loss своей trajectory. FSDP усредняет gradients; если у rank
несколько trajectories, local loss сначала усредняется локально. Metrics затем
суммируются через `accelerator.reduce`.

До первого optimizer step для всех выбранных групп фиксируются:

- behavior-policy log-probabilities с текущим adapter;
- reference log-probabilities того же NF4 base с отключённым adapter.

Adapter fingerprint запрещает обучать rollout не той starting policy. После
завершения update нужны новые rollout'ы.

Exact ATIF trajectory может содержать разное число LLM calls. Один decoder
forward на одном rank обязан иметь соответствующий forward на остальных ranks.
Максимальное число segments согласуется `all_reduce(MAX)`; недостающие calls
дополняются двухтокенными forwards. Их output умножается на ноль, но остаётся в
loss graph, поэтому backward collectives тоже вызываются одинаковое число раз.

## Monitoring

Rank 0 пишет phase telemetry каждые 5 секунд:

- NVML utilization, memory-controller utilization, VRAM, power и temperature
  для всех четырёх GPU;
- local PyTorch allocated/reserved/max-allocated;
- CPU%, RSS, system available RAM и load average;
- фазы tokenize, NF4 load, FSDP wrap, old/reference cache, forward/backward,
  optimizer, full-state gather и adapter save.

```bash
gemma-grpo telemetry --input /path/to/adapter.telemetry.jsonl
```

В отличие от старого sequential device map, во время decoder forward/backward
ожидается активность всех четырёх GPU. Короткие провалы между trajectories и во
время CPU/tokenization являются нормальными.

## Остаточные риски

### Hardware smoke ещё обязателен

CPU tests проверяют orchestration, loss и форматы, но не могут подтвердить
совместимость Gemma 4 31B + Transformers 5.0 + PEFT 0.19.1 + Accelerate 1.13 +
Torch 2.10 + BNB kernels + NCCL на L4. До одного полного Kaggle smoke новый
путь логически проверен, но не hardware-validated.

### Full-state adapter save

PEFT/FSDP требует переключиться на `FULL_STATE_DICT` перед обычным
`save_pretrained`. Это создаёт кратковременный CPU/RAM и communication peak в
конце update. Системных 173 GiB должно хватить, но telemetry должна подтвердить.

### 32k activations

Chunked vocabulary projection убирает известный пик logits, но не decoder
activations. Smoke ограничен 8192; увеличивать 12k → 16k → 32k нужно только по
фактическому запасу VRAM.

### Token reconstruction

`chat_template_reconstruction` не гарантирует token-by-token совпадение с vLLM.
Наиболее надёжны traces с `prompt_token_ids` и `completion_token_ids` каждого
assistant call.

### Final W4 evaluator

On-policy пара теперь согласована внутри NF4: rollout и trainer используют один
base. Но competition submission будет исполняться на официальном QAT W4 base.
LoRA module names совместимы, logits не bit-exact. Нужен отдельный W4 regression
eval перед отправкой; это eval, а не возврат W4 в training loop.

## Критерии успешного smoke

1. Preflight подтверждает BNB ≥0.48.1, общий NF4 checkpoint, runtime shim, встроенный BNB loader и 4 GPU.
2. vLLM запускает NF4 PP=4, возвращает `/v1/models`, собирает 4 scorable rollout'а.
3. Перед training в логе есть `[vllm] stopped`.
4. FSDP world size равен 4; каждый rank получает trajectory.
5. Завершаются old/reference cache, forward/backward и ровно один optimizer step.
6. Loss, KL и grad norm конечны; telemetry не показывает OOM/CPU offload.
7. Созданы `adapter_config.json`, `adapter_model.safetensors` и
   `training_metrics.json`.
8. Следующий managed vLLM стартует с сохранённым adapter и свежими seeds.

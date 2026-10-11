# Аудит GRPO training path на Kaggle L4 x4

Дата аудита: 2026-10-11. Целевая конфигурация: Gemma 4 31B QAT W4 checkpoint,
четыре NVIDIA L4 по 22.03 GiB, LoRA/QLoRA и SWEGemma rollouts.

## Итог

Причина последнего OOM локализована: trainer создавал полный FP32 tensor
`action_tokens × vocabulary` внутри `log_softmax`. Для наблюдавшихся 6 574
assistant-токенов и vocab 262 144 это ровно около 6.42 GiB поверх уже размещённой
модели. Это не утечка и не фрагментация allocator'а.

Код исправлен: backbone считается один раз, нужные hidden states выбираются до
`lm_head`, vocabulary projection и `log_softmax` идут chunk'ами по 32 токена,
а chunk forward пересчитывается checkpointing'ом при backward. GPU 0 дополнительно
получает 8 GiB резерва против 4 GiB на остальных GPU.

CPU unit/regression suite проходит, но полная совместимость direct backbone path,
Accelerate hooks, bitsandbytes NF4 и Gemma 4 31B должна быть подтверждена одним
hardware smoke на Kaggle. До такого прогона исправление считается проверенным
логически и на малой модели, но не подтверждённым end-to-end на L4.

## Исправленные дефекты

| Приоритет | Дефект | Исправление |
|---|---|---|
| P0 | FP32 `6574 × 262144` logits требовали 6.42 GiB на GPU 0 | Chunked `lm_head + log_softmax`, default 32, activation checkpointing |
| P0 | Прямой chunked `lm_head` мог обойти Gemma 4 `final_logit_softcapping` и исказить log-probabilities | В каждом chunk применяется точная model-specific формула до `log_softmax` |
| P0 | GPU 0 имел тот же weight budget, хотя держит embeddings/lm_head | Резерв 8 GiB на GPU 0 и 4 GiB на GPU 1–3 |
| P1 | Длинная траектория отбрасывалась отдельно, но advantage оставался нормирован по полной группе | Если один rollout слишком длинный, пропускается вся группа |
| P1 | `max_steps` применялся до tokenization и мог выбрать только непригодные группы | Лимит теперь считает пригодные группы |
| P1 | Частичные ATIF token IDs могли тихо исключить некоторые assistant turns | Exact-token режим используется только при наличии IDs у всех assistant calls |
| P1 | Rollout без trainable assistant trace мог получить reward и попасть в группу | Такой rollout помечается `missing_trainable_trace`, `scorable=false` |
| P1 | NaN/Inf objective или gradient мог попасть в optimizer | Явные finite-checks до `backward`/`step` |
| P2 | Ошибки длины обнаруживались после пятиминутной загрузки 31B | Rollouts токенизируются и проверяются до model load |
| P2 | Повторный запуск мог затереть telemetry падения | Следующий файл получает суффикс `-v2`, `-v3`, ... |

## Почему не загружены все четыре GPU одновременно

`infer_auto_device_map + dispatch_model` раскладывает последовательные decoder
layers по GPU. Один microbatch сначала исполняет ранние layers на GPU 0, затем
передаёт activations на GPU 1, GPU 2 и GPU 3. Это pipeline-like model placement
без pipeline microbatch schedule, а не tensor parallel и не data parallel.

Дополнительно:

- NF4 requantization намеренно использует GPU 0 как scratch device;
- embeddings и `lm_head` находятся на GPU 0;
- batch фактически равен одной траектории, потому что 6–8k token agent traces
  обрабатываются последовательно ради памяти.

Поэтому граф utilization будет пилообразным. Добиваться одновременных 100% на
всех L4 настройкой batch size в текущей архитектуре нельзя. Для этого нужен
отдельный trainer на FSDP/DeepSpeed или tensor/pipeline parallel с несколькими
microbatch, а также иной способ сосуществования с rollout server.

## Что измеряется

`TrainingMonitor` пишет sample каждые 5 секунд и немедленно при смене фазы:

- exact phase и elapsed time;
- GPU compute utilization и memory-controller utilization;
- NVML VRAM used/total, power и temperature для всех устройств;
- process-local PyTorch allocated/reserved/max-allocated для каждого GPU;
- process CPU%, CPU time, RSS и threads;
- system available RAM и load average.

Фазы включают `load_checkpoint_cpu`, `requantize_nf4`, `prepare_kbit`,
`inject_lora`/`load_lora`, `infer_device_map`, `dispatch_model`, `cache_old`,
`cache_reference`, `forward_backward`, `optimizer` и `save_adapter`.

Агрегация:

```bash
gemma-grpo telemetry --input /path/to/adapter.telemetry.jsonl
```

Вывод содержит и точные фазы с group/seed, и `phase_kinds`, где все seed одной
операции объединены.

## Оставшиеся риски

### 1. Rollout policy и trainer policy не bit-exact (высокий)

vLLM генерирует исходным QAT W4 checkpoint, а trainer распаковывает его и
переквантовывает в bitsandbytes NF4. `old_logprobs` пересчитываются NF4-моделью,
а не сохраняются непосредственно во время sampling. Fingerprint адаптера
проверяет LoRA policy, но не устраняет расхождение base kernels/quantization.

Для строго on-policy GRPO нужно либо сохранять выбранные token log-probabilities
из rollout server, либо считать rollout и backward одной и той же bit-exact
моделью. Текущий путь — практическое приближение.

### 2. Текущие smoke rollouts реконструированы chat template (высокий)

В наблюдавшихся строках `token_source=chat_template_reconstruction`, а tools
definitions отсутствуют. Prefix preservation проверена, но это не доказывает
совпадение токенов с vLLM. Для длинного обучения нужны полные ATIF
`prompt_token_ids`/`completion_token_ids` для каждого assistant call либо
отдельная валидация token-by-token против serving request.

### 3. Accelerate device map — не полноценный distributed trainer (средний)

Autograd через dispatch hooks должен быть подтверждён hardware smoke. Даже после
успеха это решение годится для эксперимента, но не даёт равномерного throughput.

### 4. 32k context пока не подтверждён (средний)

Chunked vocabulary projection убирает конкретный 6.42 GiB пик, но не память
transformer activations. Smoke traces имеют примерно 6.5–7.7k токенов. Длину
следует повышать 8k → 12k → 16k → 32k только по telemetry.

### 5. Автоматический loop не владеет внешним vLLM server (высокий для local models.yaml)

Если `models.yaml` указывает на вручную поднятый `127.0.0.1:8000`, команда
`gemma-grpo loop` не умеет сама остановить этот чужой процесс перед backward и
перезапустить его после adapter update. В таком режиме обещание полностью
автоматического collect/train неверно: нужен managed server lifecycle. Без
`models.yaml` это зависит от того, стартует ли конкретная версия competition
harness server на время каждого `swegemma eval` и освобождает ли его после.

До реализации managed lifecycle не запускать 100-step loop с оставленным
локальным vLLM — он заберёт VRAM у trainer.

### 6. Дорогая перезагрузка между update (средний)

Каждый train update заново распаковывает 31B и делает NF4 requantization. Это
необходимо при переиспользовании тех же GPU для vLLM и backward, но дорого.
Собирайте несколько задач/групп перед одним update, чтобы амортизировать загрузку.

## Критерии успешного smoke

1. В логе есть четыре GPU weight-memory budget и device map без `cpu`/`disk`.
2. Обе траектории проходят tokenization до model load.
3. `cache_old`, `cache_reference` и оба `forward_backward` завершаются.
4. `loss`, `kl`, `grad_norm` конечны; выполнен ровно один optimizer step.
5. Созданы `adapter_config.json`, `adapter_model.safetensors` и
   `training_metrics.json`.
6. Telemetry заканчивается `complete`; ни один GPU не упирается устойчиво в
   полный объём VRAM.

Если OOM остаётся именно в vocabulary projection, уменьшить
`logprob_chunk_size: 32` до 16. Если OOM возникает внутри backbone, chunk size
не поможет: уменьшить `max_seq_length` или переходить на sharded training.

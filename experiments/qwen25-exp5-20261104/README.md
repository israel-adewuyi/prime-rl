# Exp5: Qwen2.5 placement and shared-GPU timing

Model: Qwen/Qwen2.5-0.5B-Instruct, revision 7ae557604adf67be50417f59c2c2f167def9a775.
Each run targets 100 optimizer updates with batch256, group16, LR1e-5, 16K trainer/orchestrator packing, head chunk8192, 128 requests per engine, and 128-token train/eval completions. Use CK FlashAttention, BF16 computation/inference and the unchanged optimizer/reduction defaults. No LoRA or gradient capture. Each engine has an explicit 8GiB KV cache; this overrides automatic percentage-based cache sizing.

The shared base.toml and CONDITIONS in run.py define six trials: AlphabetSort solo on GPU0; AlphabetSort with inference on GPU1 and trainer on GPU2; and two AlphabetSort plus two ReverseText runs sharing GPU3. The solo and paired controls use inference seed42; each shared task uses seeds42 and43. These are explicit launch selections, not an automatically advancing queue. Override physical assignments with --gpus if needed.

Launch alphabet-solo first. Start all other conditions only after the user requests them. Prepare every condition on CPU with --dry-run, using a separate output root from real training. The launcher requires the pinned model in the existing Hugging Face cache and a clean tracked source tree. Each launch records its commit, model revision, resolved configuration, commands, ports and physical GPU assignment.

```bash
uv --no-config run --no-project --active python experiments/qwen25-exp5-20261104/run.py \
  --condition alphabet-solo --output-dir /mnt/shared/.research-session/exp5-20261104/dry-runs --dry-run

# After GPU allocation and the requested launch decision:
uv --no-config run --no-project --active python experiments/qwen25-exp5-20261104/run.py \
  --condition alphabet-solo --output-dir /mnt/shared/.research-session/exp5-20261104/runs
```

Every run profiles optimizer step21 with the existing trainer.trace_step option. Traces are separate per run; no profiler code changes are required. Mark profiled intervals when interpreting other co-resident runs as well as the profiled run itself. Only alphabet-solo saves standard trainer/orchestrator checkpoints at steps10,20,...,100. Export model weights with tools/convert_dcp_to_bf16.py and upload the exported HF-format directory; optimizer state is not part of that export. The current converter requires a GPU, so finish export before releasing compute. Keep the resolved run configuration with checkpoints until export is verified.

Use built-in AlphabetSort and ReverseText tasks/rewards. Evaluation selects the first32 task indices, and training excludes them; this is an episode-index split, not a guarantee that AlphabetSort author names never overlap. Evaluation runs initially, every10 updates and at completion. Checkpoint/evaluation/profile costs must be identified separately and remain part of total launch-to-completion time.

Stage the generated subprocess harness environment on node-local storage before rollout startup. Preserve source, model cache and requested results on /mnt/shared. Monitor CPU quota/throttling, process and I/O pressure, GPU activity/memory, generated and admitted rollouts, no-signal rejection, batch waiting, forward/backward and weight refresh. A trainer trace does not include inference/harness activity in their other processes. Report per-run elapsed time and aggregate completed updates per GPU-hour, including startup and separate later-step statistics; do not interpret GPU busy time as FLOP utilization.

The launcher does not provision GPUs, change bids, upload to HF or start another condition automatically.

## Rollout supply diagnostic

Three 30-update AlphabetSort probes use GPUs0,1,2 respectively: `supply-control` keeps the 128-episode cap and default16 admission burst; `supply-refill128` uses cap128/burst128; `supply-pipeline256` uses cap256/burst256. The engine stays at128 sequences in all three. All use seed42, B256/group16 and the existing base settings, including eager inference, filesystem refresh and built-in rewards. No new checkpoints. Step21 is profiled and excluded from ordinary timing summaries. The absolute rental cutoff remains authoritative.

Pipeline gauges sample every second: admission budget, free episode permits, policy-update pause, train-dispatch gate and output queue, alongside existing running/waiting inference requests, episode phase timings, rejection and CPU metrics. These are snapshots, not exact elapsed blocking durations. One in-flight episode can spend time in harness/setup/scoring rather than an engine request; the engine and orchestrator limits have different meanings.

`orchestrator.concurrency.admission_burst` is optional and preserves previous behavior when omitted. The five-second window meters net expansion; natural completions refund admissions. This probe does not disable zero-advantage pruning, change group size, alter sampling, or relax policy-version gates. Compare useful admitted rollouts and completed updates, not only GPU request occupancy. Separate RCCL diagnostic ports are also recorded for these probes.

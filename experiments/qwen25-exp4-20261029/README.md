# Exp4: Qwen2.5 on AMD

Source branch: feat/prompt-gradient-capture. Keep main separate.
Model: Qwen/Qwen2.5-0.5B-Instruct at 7ae557604adf67be50417f59c2c2f167def9a775.

The shared base and run.py define three tracks: AlphabetSort, AlphabetSort with per-rollout gradient capture, and reverse-text. Both tasks use the built-in Verifiers reward and subprocess harness. No LoRA. Computation/inference remain BF16 and optimizer/reduction defaults remain unchanged. Train/eval completions are limited to 128 tokens; inference context is 4096 and trainer/orchestrator packing is 16384. The LM-head chunk is 8192. Inference uses a 10% GPU-memory budget per engine, with no explicit KV-byte override.

Prioritize matched short pilots before larger queues: B32/group16, LR1e-5, ten optimizer steps, 128 requests per engine. Compare one physical GPU hosting both components against an inference/trainer pair, then 1/2/4 independent jobs per pair using disjoint listener slots. Repeat with inference seeds 42 and 43 if time allows. Aim for 10-20 trials per track only if the measured speed allows it within the rental; do not sacrifice core comparisons to fill a queue. The script exposes batch size, LR, steps, requests and head chunk for clearly logged follow-ups. A 200-step ordinary continuation may use remaining time.

Use --gpus 0 for colocation or --gpus 0,1 for a pair. Both use the existing external-inference launch path; physical GPU masks are the only placement difference. Reserve slots 0-19. Router/backend ports occupy 4000-4951, router metrics 25000-25950, engine RPC 18000-18950 and rollout transport 21000-21950. Each trial has its own output directory and inference process. Dry runs validate and emit component configurations without starting engines or rollouts.

Prepare the generated harness script environment in a GPU-local UV_CACHE_DIR before any rollouts start. The main code, model cache and requested output evidence may stay on shared storage. Record startup separately; report later step timings and aggregate updates/GPU-hour. Capture's per-rollout statistics and verification are deliberately part of its timing cost. Keep metrics and rollout/index files needed by the dashboard. No checkpoints are enabled by default.

Use --profile-step 3 on one ordinary run to record a single complete trainer update after two warmup updates. The trace is exported immediately after that step. Exclude that update from ordinary timing estimates.

## Architecture support

vLLM supplies Qwen2/Qwen2.5 inference. The PrimeRL registry separately resolves model_type=qwen2 to the packed Qwen2ForCausalLM adapter. It reuses the dense Llama backbone while selecting biased Q/K/V projections, unbiased output/MLP projections, no Q/K normalization, and Qwen2's per-layer sliding-window configuration. PyTorch autograd and the shared attention/MLP kernels provide backward computation; this is not an alias to Qwen3. State-dict names/shapes and tied embeddings match Hugging Face Qwen2. Nonzero attention dropout is rejected.

The CPU parity test substitutes only the attention kernel with SDPA, so it validates architecture and packing, not the GPU FlashAttention backend. Run the GPU-marked portability and isolation checks on the target runtime before claiming GPU correctness. Strict numerical failures must be reported without loosening tolerances.
# LFM learning-rate sweep: 2026-10-23

This directory records the executed exp2 settings after the run. It is not the
commit from which that run was launched: execution used base `57c22db` plus an
uncommitted worktree and two source repairs. Do not label historical results with
this new commit as though it existed at execution time.

The ten-rate grid is `{1, 5} × {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}`, with 200 updates
per sweep. The 500- and 1,000-update repeats use 1e-6. A separate five-update
configuration captures and verifies prompt gradients. All configurations use
LiquidAI/LFM2.5-230M, BF16 compute/inference and unchanged FP32 optimization and
reduction defaults, sequence cap 256, response cap 64, batch 8 and group size 4.

`rewards.py` intentionally preserves the historical custom training reward and
exact-match evaluation scorer. The ordinary gradient-capture example uses the
built-in reverse-text reward. These historical configs must not silently change
their objective when the ordinary example changes.

Evaluation uses the first 32 task indices, excluded from training, every 50
updates. The model snapshot observed at execution and SHA-256 hashes of the
original executed TOMLs are in `manifest.json`. Checked-in TOMLs differ only in
portable output paths and the location of their equivalent reward functions.
The model snapshot is recorded provenance; the original TOMLs did not pin it.

From the repository root, in the provisioned ROCm environment, a single run is:

```bash
HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 \
  uv --no-config run --no-project --active rl \
  @ experiments/lfm-lr-sweep-20261023/configs/lfm-lr-1e-06-200.toml
```

The historical run plan assigned six concurrent jobs to each of GPU pairs 0/1
and 2/3; capture used 4/5. Configs retain separate service/transport ports and run
names. GPU assignments in the manifest are deployment metadata, not allocations
made by opening or parsing this directory.

Only LR 1e-5 and 1e-7 completed 200 updates; the rest remained partial at the
one-hour rental cutoff. All recorded held-out exact-match evaluations were
0/32. The 500/1,000-update repeats reached 172/191, and all five capture checks
passed. Full logs and the interactive report remain outside source control in
the sibling `research_manager/research_outcomes/exp2-20261023/` directory.
No raw logs, checkpoint weights or gradient tensors are stored here.

"""Profile the real trainer head dimensions and FP64 capture accumulation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import time
import tomllib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    assert os.environ["HIP_VISIBLE_DEVICES"] == os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    import torch
    from prime_rl.trainer.models.layers.lm_head import _SequenceChunkedLogProbEntropyFn, _online_logsumexp_and_weighted_update
    from prime_rl.trainer.rl.gradient_statistics import CUDAGradientStatistics
    assert torch.cuda.device_count() == 1
    cfg = tomllib.loads(Path(__file__).with_suffix(".toml").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg["seed"])
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    print(json.dumps(dict(config=cfg, commit=sha, mask="3", visible=1, torch=torch.__version__, hip=torch.version.hip)), flush=True)
    result = {"config": cfg, "commit": sha, "timings": {}}
    def bench(name, fn):
        for _ in range(cfg["warmup"]): fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(cfg["repeats"]):
            torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
            samples.append(time.perf_counter()-t)
        result["timings"][name] = {"seconds": samples, "median": statistics.median(samples)}
        print(name, result["timings"][name], flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            fn(); torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output_dir / (name+".json")))
        (args.output_dir / (name+".txt")).write_text(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=30))
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    logits = torch.randn((cfg["token_chunk"], cfg["vocab_chunk"]), device="cuda", dtype=torch.float32)
    m = torch.zeros(cfg["token_chunk"], device="cuda"); s = torch.ones_like(m); t = torch.zeros_like(m)
    bench("online_update", lambda: _online_logsumexp_and_weighted_update(m,s,t,logits))
    del logits,m,s,t
    h = torch.randn((cfg["tokens"], cfg["hidden"]), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((cfg["vocab"], cfg["hidden"]), device="cuda", dtype=torch.bfloat16, requires_grad=True) * 0.02
    w = w.detach().requires_grad_()
    labels = torch.randint(cfg["vocab"], (cfg["tokens"],), device="cuda")
    inv = torch.full((cfg["tokens"],), 1/cfg["temperature"], device="cuda")
    upstream = torch.randn(cfg["tokens"], device="cuda") / cfg["tokens"]
    def head():
        lp, ent = _SequenceChunkedLogProbEntropyFn.apply(h,w,labels,inv,cfg["token_chunk"],None)
        lp.backward(upstream); h.grad = None; w.grad = None
    bench("head_forward_backward", head)
    del h,w,labels,inv,upstream
    header = {}
    for f in Path(cfg["model"]).glob("*.safetensors"):
        with f.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header.update(json.loads(stream.read(size)))
    shapes = {name: rec["shape"] for name,rec in header.items() if name != "__metadata__"}
    assert shapes
    gradients = {name: torch.randn(shape, device="cuda", dtype=torch.float32) * 0.001 for name,shape in shapes.items()}
    print("GRADIENT_SHAPES",json.dumps(shapes),flush=True)
    stats = CUDAGradientStatistics()
    bench("statistics_update", lambda: stats.update(gradients))
    print("STATISTICS_METRICS", stats.finalize(),flush=True)
    print("COMPLETE",sha,flush=True)


if __name__ == "__main__":
    main()

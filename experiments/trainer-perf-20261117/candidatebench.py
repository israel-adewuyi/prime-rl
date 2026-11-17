"""Matched alternating warmed benchmarks and strict numerical checks of the prototype."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import tomllib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    assert os.environ["HIP_VISIBLE_DEVICES"] == os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    import torch
    import prime_rl.trainer.models.layers.lm_head as head
    from online_kernel import online_update
    assert torch.cuda.device_count() == 1
    cfg = tomllib.loads(Path(__file__).with_name("microbench.toml").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    original = head._online_logsumexp_and_weighted_update
    variants = {"baseline": original, "triton_online_update": online_update}
    result = dict(commit=sha, config=cfg, correctness={}, timings={})
    print("CONFIG",json.dumps(result),"MASK 3 VISIBLE 1",flush=True)
    torch.manual_seed(cfg["seed"])
    def check(name,a,b):
        diff = (a.float()-b.float()).abs()
        result["correctness"][name] = dict(max_abs=diff.max().item(), rel_l2=(diff.norm()/a.float().norm().clamp_min(1e-30)).item())
        print("CORRECTNESS",name,result["correctness"][name],flush=True)
        torch.testing.assert_close(a,b,rtol=cfg["correctness_rtol"],atol=cfg["correctness_atol"])
    def paired(name,functions):
        for _ in range(cfg["warmup"]):
            for fn in functions.values():fn()
        samples = {key:[] for key in functions}
        for repeat in range(cfg["repeats"]):
            keys = list(functions) if repeat%2==0 else list(reversed(functions))
            for key in keys:
                torch.cuda.synchronize(); start=time.perf_counter();functions[key]();torch.cuda.synchronize()
                samples[key].append(time.perf_counter()-start)
        result["timings"][name] = {key:dict(seconds=vals,median=statistics.median(vals),min=min(vals),max=max(vals)) for key,vals in samples.items()}
        print("TIMINGS",name,json.dumps(result["timings"][name]),flush=True)
        for key,fn in functions.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                fn();torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output_dir/(name+"-"+key+".json")))
        (args.output_dir/"results.json").write_text(json.dumps(result,indent=2))
    x=torch.randn((cfg["token_chunk"],cfg["vocab_chunk"]),device="cuda")
    m=torch.randn(cfg["token_chunk"],device="cuda");s=torch.rand_like(m);t=torch.rand_like(m)
    a=original(m,s,t,x);b=online_update(m,s,t,x)
    for key,aa,bb in zip(["m","s","t"],a,b):
        # Raw partition sums scale with vocabulary; test normalized logZ and entropy.
        if key=="m":check("helper_max",aa,bb)
    check("helper_logz",a[0]+a[1].log(),b[0]+b[1].log())
    check("helper_entropy",a[0]+a[1].log()-a[2]/a[1],b[0]+b[1].log()-b[2]/b[1])
    paired("online_update",{key:lambda fn=fn:fn(m,s,t,x) for key,fn in variants.items()})
    del x,m,s,t,a,b
    h=torch.randn((cfg["tokens"],cfg["hidden"]),device="cuda",dtype=torch.bfloat16,requires_grad=True)
    w=(torch.randn((cfg["vocab"],cfg["hidden"]),device="cuda",dtype=torch.bfloat16)*.02).requires_grad_()
    labels=torch.randint(cfg["vocab"],(cfg["tokens"],),device="cuda")
    inv=torch.full((cfg["tokens"],),1/cfg["temperature"],device="cuda")
    up=torch.randn(cfg["tokens"],device="cuda")/cfg["tokens"]
    def run(fn,save=False):
        head._online_logsumexp_and_weighted_update=fn
        lp,ent=head._SequenceChunkedLogProbEntropyFn.apply(h,w,labels,inv,cfg["token_chunk"],None)
        loss=(lp*up).sum();loss.backward()
        out=(lp.detach(),ent.detach(),loss.detach(),h.grad.clone(),w.grad.clone()) if save else None
        h.grad=None;w.grad=None
        return out
    a=run(original,True);b=run(online_update,True)
    for key,aa,bb in zip(["logprobs","entropy","loss","hidden_grad","weight_grad"],a,b):check(key,aa,bb)
    del a,b
    paired("head_forward_backward",{key:lambda fn=fn:run(fn) for key,fn in variants.items()})
    head._online_logsumexp_and_weighted_update=original
    print("COMPLETE",sha,flush=True)


if __name__=="__main__":
    main()

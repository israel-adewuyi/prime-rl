"""Run one explicitly declared kernel variant in the synthetic trainer control."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tomllib


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output_dir",type=Path)
    parser.add_argument("implementation",choices=["baseline","triton_online_update"])
    args=parser.parse_args()
    assert os.environ["HIP_VISIBLE_DEVICES"]==os.environ["CUDA_VISIBLE_DEVICES"]=="3"
    import torch
    assert torch.cuda.device_count()==1
    here=Path(__file__).resolve().parent
    sweep=tomllib.loads((here/"microbench.toml").read_text())
    assert args.implementation in sweep["implementations"]
    torch.manual_seed(sweep["seed"])
    sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"])
    raw=tomllib.loads((here/"compute-profile.toml").read_text())
    raw.update(output_dir=str(args.output_dir),trace_path=str(args.output_dir/"profile"))
    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.utils.config import dump_resolved_config
    from prime_rl.trainer.rl.train import train
    if args.implementation=="triton_online_update":
        import prime_rl.trainer.models.layers.lm_head as head
        from online_kernel import online_update
        head._online_logsumexp_and_weighted_update=online_update
    args.output_dir.mkdir(parents=True,exist_ok=False)
    config=TrainerConfig.model_validate(raw)
    (args.output_dir/"resolved-config.json").write_text(json.dumps(dump_resolved_config(config),indent=2))
    print("LAUNCH",json.dumps(dict(commit=sha,implementation=args.implementation,argv=os.sys.argv,mask="3",visible=1,seed=sweep["seed"],affinity=sorted(os.sched_getaffinity(0)))),flush=True)
    train(config)


if __name__=="__main__":
    main()

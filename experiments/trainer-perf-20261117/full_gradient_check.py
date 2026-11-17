"""Compare pre-update FP32 model gradients on one fixed synthetic packed batch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tomllib


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output_dir",type=Path)
    args=parser.parse_args()
    assert os.environ["HIP_VISIBLE_DEVICES"]==os.environ["CUDA_VISIBLE_DEVICES"]=="3"
    import torch
    assert torch.cuda.device_count()==1
    from prime_rl.trainer.rl.data import FakeDataLoader
    import prime_rl.trainer.models.layers.lm_head as head
    from online_kernel import online_update
    from prime_rl.trainer.rl.train import train
    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.utils.config import dump_resolved_config
    torch.manual_seed(42)
    here=Path(__file__).resolve().parent
    cfg=tomllib.loads((here/"correctness.toml").read_text())
    cfg["output_dir"]=str(args.output_dir)
    config=TrainerConfig.model_validate(cfg)
    sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"resolved-config.json").write_text(json.dumps(dump_resolved_config(config),indent=2))
    original_batch=FakeDataLoader.get_batch
    cached=None
    def get_batch(self):
        nonlocal cached
        if cached is None:cached=original_batch(self)
        else:head._online_logsumexp_and_weighted_update=online_update
        return cached
    FakeDataLoader.get_batch=get_batch
    previous=[]
    result=dict(commit=sha,mask="3",visible=1,optimizer_updates=0,gradient_rtol=0.,gradient_atol=1e-6,parameters=[])
    def step(self,closure=None):
        params=[p for group in self.param_groups for p in group["params"] if p.grad is not None]
        local=lambda x:x.to_local() if hasattr(x,"to_local") else x
        if not previous:
            previous.extend(local(p.grad).detach().clone() for p in params)
            print("BASELINE_GRADIENTS",len(previous),flush=True)
        else:
            assert len(params)==len(previous)
            for index,(p,reference) in enumerate(zip(params,previous)):
                actual=local(p.grad)
                diff=(actual-reference).abs()
                result["parameters"].append(dict(index=index,shape=list(actual.shape),max_abs=diff.max().item(),rel_l2=(diff.norm()/reference.norm().clamp_min(1e-30)).item()))
                (args.output_dir/"gradient-check.json").write_text(json.dumps(result,indent=2))
                torch.testing.assert_close(actual,reference,rtol=0.,atol=1e-6)
            print("FULL_GRADIENT_PASS",len(params),"max_abs",max(x["max_abs"] for x in result["parameters"]),flush=True)
        return None
    torch.optim.AdamW.step=step
    print("LAUNCH",json.dumps(dict(commit=sha,mask="3",visible=1,argv=os.sys.argv,config=cfg)),flush=True)
    train(config)
    rows=[json.loads(line) for line in (args.output_dir/"monitors/file/metrics.jsonl").open()]
    losses=[row["loss/mean"] for row in rows if "loss/mean" in row]
    assert len(losses)==2
    result["losses"]=losses
    assert abs(losses[0]-losses[1])<=1e-6
    result["pass"]=True
    (args.output_dir/"gradient-check.json").write_text(json.dumps(result,indent=2))
    print("COMPLETE",sha,flush=True)


if __name__=="__main__":
    main()

"""Exp3.2: SmolLM2 Instruct batch-size/LR sweep and gradient capture."""
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
RATES = ['1e-7', '5e-7', '1e-6', '5e-6', '1e-5', '5e-5']
RUNS = [(batch, lr, 200, False) for batch in (256, 512) for lr in RATES]
RUNS += [(256, '1e-6', 500, False), (256, '1e-6', 1000, False), (256, '1e-6', 5, True)]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    p.add_argument('--index', type=int, choices=range(len(RUNS)))
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--gpus', default='0,1', help='Inference GPU,trainer GPU')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/exp3.2-20261025'))
    p.add_argument('--steps', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--pack', type=int, default=16384)
    p.add_argument('--requests-per-engine', type=int, choices=(128, 256, 512),
                   help='Set vLLM sequence capacity and fixed orchestrator in-flight episodes together')
    p.add_argument('--slot', type=int, help='Unique listener slot; all ports below ephemeral range')
    p.add_argument('--name')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.list:
        for i, (batch, lr, steps, capture) in enumerate(RUNS):
            print(json.dumps(dict(index=i,batch_size=batch,lr=lr,steps=steps,capture=capture)))
        return
    if args.index is None:
        p.error('--index is required')
    gpu = args.gpus.split(',')
    if len(gpu) != 2 or not all(g.isdigit() for g in gpu) or gpu[0] == gpu[1]:
        p.error('--gpus requires two distinct ordinals')
    batch, lr, steps, capture = RUNS[args.index]
    batch = args.batch_size or batch
    steps = args.steps or steps
    if batch % 16 or steps < 1 or args.pack < 512 or (capture and steps != 5):
        p.error('batch must be divisible by 16; pack >=512; capture requires five steps')
    slot = args.slot if args.slot is not None else args.index
    if not 0 <= slot <= 29:
        p.error('slot must be 0..29')
    name = args.name or ('smol-gradient-capture-5' if capture else f'smol-b{batch}-lr{lr}-{steps}')
    repo = args.repo.resolve()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip():
        p.error('Commit tracked source changes before launching')
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(MODEL,revision=REVISION,local_files_only=True)
    port = 4000 + slot*100
    cmd = ['uv','--no-config','run','--no-project','--active','rl','@',str(Path(__file__).with_name('base.toml')),
           '--model.name',model_path,'--run.name',name,'--output-dir',str(args.output_dir.resolve()),
           '--max-steps',str(steps),'--trainer.optim.lr',lr,'--orchestrator.batch-size',str(batch),
           '--trainer.model.seq-len',str(args.pack),'--orchestrator.seq-len',str(args.pack),
           '--inference.server.port',str(port),'--inference.backend-port',str(port+1),
           '--inference.vllm.data-parallel-rpc-port',str(26000+slot*100),
           '--orchestrator.model.client.base-url',f'http://127.0.0.1:{port}/v1',
           '--orchestrator.model.client.admin-base-url',json.dumps([f'http://127.0.0.1:{port+1}/v1']),
           '--rollout-transport.port',str(20000+slot*100),'--no-dashboard']
    if capture:
        cmd += ['--trainer.gradient-capture.steps','[1,2,3,4,5]','--trainer.gradient-capture.verify',
                '--ckpt.interval','5','--ckpt.keep-last','1','--ckpt.save-gradients-and-updates']
    if args.requests_per_engine is not None:
        limit = str(args.requests_per_engine)
        cmd += ['--inference.vllm.max-num-seqs', limit,
                '--inference.vllm.kv-cache-memory-bytes', str(8 * 1024**3),
                '--orchestrator.concurrency.initial-inflight', limit,
                '--orchestrator.concurrency.min-inflight', limit,
                '--orchestrator.concurrency.max-inflight', limit]
    if args.dry_run:
        cmd += ['--dry-run']
    record = dict(commit=commit,model=MODEL,model_revision=REVISION,index=args.index,name=name,
                  lr=lr,batch_size=batch,steps=steps,capture=capture,pack=args.pack,gpus=args.gpus,slot=slot,
                  requests_per_engine=args.requests_per_engine,
                  kv_cache_gib=8 if args.requests_per_engine is not None else 1,command=cmd)
    print(json.dumps(record),flush=True)
    print(shlex.join(cmd),flush=True)
    env = dict(os.environ,HIP_VISIBLE_DEVICES=args.gpus,CUDA_VISIBLE_DEVICES=args.gpus)
    raise SystemExit(subprocess.call(cmd,cwd=repo,env=env))

if __name__ == '__main__':
    main()

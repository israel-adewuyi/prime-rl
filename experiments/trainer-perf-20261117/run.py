"""Launch the committed isolated trainer baseline; runtime wiring only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tomllib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert os.environ["HIP_VISIBLE_DEVICES"] == os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    import torch
    assert torch.cuda.device_count() == 1
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    raw = tomllib.loads((here / "rl.toml").read_text())
    raw.update(output_dir=str(args.output_dir), dry_run=args.dry_run)
    raw["trainer"]["trace_path"] = str(args.output_dir / "normal-step3.json")
    from prime_rl.configs.rl import RLConfig
    import prime_rl.entrypoints.rl as entry
    config = RLConfig.model_validate(raw)
    print(json.dumps(dict(commit=sha, mask="3", visible=torch.cuda.device_count(),
                         affinity=sorted(os.sched_getaffinity(0)), torch=torch.__version__,
                         hip=torch.version.hip, argv=os.sys.argv)), flush=True)
    original = entry.torchrun_cmd
    def command(*a, **kw):
        cmd = original(*a, **kw)
        cmd = ["--rdzv-endpoint=localhost:29681" if x.startswith("--rdzv-endpoint=") else x for x in cmd]
        print("TRAINER_LAUNCH", json.dumps(cmd), flush=True)
        return cmd
    entry.torchrun_cmd = command
    entry.rl(config)


if __name__ == "__main__":
    main()

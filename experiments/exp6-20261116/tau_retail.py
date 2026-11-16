"""Run 10 retail tasks, three trials, with local Qwen in every LLM role."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path('/mnt/shared/tau2-bench'))
    parser.add_argument('--api-base',default='http://127.0.0.1:8300/v1')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    llm_args=dict(api_base=args.api_base,api_key='dummy',temperature=0,max_tokens=2048,
        extra_body=dict(chat_template_kwargs=dict(enable_thinking=False)))
    model='hosted_vllm/retail-agent'
    command=['uv','run','--no-sync','tau2','run','--domain','retail',
        '--agent-llm',model,'--agent-llm-args',json.dumps(llm_args),
        '--user-llm',model,'--user-llm-args',json.dumps(llm_args),
        '--communicate-judge',json.dumps(dict(model=model,model_args=llm_args)),
        '--num-tasks','10','--num-trials','3','--max-concurrency','3',
        '--seed','300','--verbose-logs','--llm-log-mode','all',
        '--save-to',str(args.output_dir.resolve())]
    if args.dry_run:
        command+=['--dry-run']
    else:
        print('Waiting for the final Qwen endpoint with retail-agent alias',flush=True)
        while True:
            try:
                with urllib.request.urlopen(args.api_base+'/models',timeout=5) as response:
                    models=json.load(response)
                if any(m['id']=='retail-agent' for m in models['data']):
                    break
            except (urllib.error.URLError,TimeoutError):
                pass
            time.sleep(10)
    print('tau-bench source:',subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip(),flush=True)
    print(shlex.join(command),flush=True)
    env=dict(os.environ,UV_FROZEN='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    raise SystemExit(subprocess.call(command,cwd=args.repo,env=env))


if __name__=='__main__':
    main()

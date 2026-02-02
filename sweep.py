import subprocess
import tomli
import tomli_w
from pathlib import Path
import time

# Define your learning rate sweep

learning_rates = [1e-5]

masks = ["4940162random0"]
sparsities = ['99', ]

# Base config paths
base_train = "configs/alphabet_sort/rl/train.toml"
base_orch = "configs/alphabet_sort/rl/orch.toml"
infer_config = "configs/alphabet_sort/rl/infer.toml"  # This stays unchanged

# Directory for modified configs
sweep_dir = Path("configs/gsm8k/sweep")
sweep_dir.mkdir(exist_ok=True, parents=True)

def format_lr(lr):
    """Format learning rate for filenames (e.g., 1e-7 -> '1e7')"""
    return f"{lr:.0e}".replace('-', '')

for mask, sp in zip(masks, sparsities):
    for lr in learning_rates:
        lr_str = format_lr(lr)
        
        # Load configs (tomli reads in binary mode)
        with open(base_train, 'rb') as f:
            train_config = tomli.load(f)
        with open(base_orch, 'rb') as f:
            orch_config = tomli.load(f)
        
        # Update learning rate in train.toml
        train_config['optim']['lr'] = lr
        # train_config['load_mask']['num_active'] = mask
        
        # Update wandb names to include new learning rate
        train_config['wandb']['name'] = f"train_alphabetsort-qwen0.5B_sparsity-{sp}_lr={lr_str}_{mask}_demo"
        orch_config['wandb']['name'] = f"orch_alphabetsort-qwen0.5B_sparsity-{sp}_lr={lr_str}_{mask}_demo"
        orch_config['seq_len'] = 6144
        
        # Save modified configs (tomli_w writes in binary mode)
        train_path = sweep_dir / f"train_alphabetsort-qwen0.5B_sparsity-{sp}_lr{lr_str}_{mask}.toml"
        orch_path = sweep_dir / f"orch_alphabetsort-qwen0.5B_sparsity-{sp}_lr{lr_str}_{mask}.toml"
        
        train_config["max_steps"] = 2
        orch_config["max_steps"] = 2
        orch_config["batch_size"] = 64
        orch_config["eval"]["num_examples"] = 64
        
        with open(train_path, 'wb') as f:
            tomli_w.dump(train_config, f)
        with open(orch_path, 'wb') as f:
            tomli_w.dump(orch_config, f)
        
        # Launch training
        cmd = f"""uv run rl \
        --trainer @ {train_path} \
        --orchestrator @ {orch_path} \
        --inference @ {infer_config} \
        --trainer-gpu-ids 2 \
        --inference-gpu-ids 2 \
        --inference.gpu-memory-utilization 0.4 \
        --log.level debug"""
        
        print(f"\n{'='*60}")
        print(f"Starting experiment with lr={lr} ({lr_str})")
        print(f"{'='*60}\n")
        
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode != 0:
            print(f"\n⚠️  Experiment lr={lr} failed with code {result.returncode}")
            # break  # Uncomment to stop on first failure
        else:
            print(f"\n✓ Experiment lr={lr} completed successfully")
        
        time.sleep(10)

print("\n🎉 All experiments completed!")
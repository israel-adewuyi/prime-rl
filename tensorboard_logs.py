from tensorboard.backend.event_processing import event_accumulator
import os, pandas as pd

logdir = "outputs/runs"  # same as you pass to tensorboard --logdir
rows = []

for run in os.listdir(logdir):
    run_dir = os.path.join(logdir, run)
    if not os.path.isdir(run_dir):
        continue
    ea = event_accumulator.EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        for e in ea.Scalars(tag):
            rows.append({
                "run": run,
                "tag": tag,
                "step": e.step,
                "wall_time": e.wall_time,
                "value": e.value,
            })

df = pd.DataFrame(rows)
df.to_csv("all_tb_scalars.csv", index=False)

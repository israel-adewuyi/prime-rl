import plotly.graph_objects as go
import json
import torch
import plotly.express as px

def visualize_sparsity(json_path: str = "test.json", out_dir: str = "plots"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    with open(json_path, "r") as f:
        data = json.load(f)

    checkpoints = list(data.keys())

    # --- 1. Global sparsity across checkpoints ---
    global_vals = [data[ckpt]["global"] for ckpt in checkpoints]
    fig_global = go.Figure()
    fig_global.add_trace(go.Scatter(
        x=checkpoints,
        y=global_vals,
        mode="lines+markers",
        name="Global Update Sparsity"
    ))
    fig_global.update_layout(
        title="Global Update Sparsity Across Checkpoints",
        xaxis_title="Checkpoint",
        yaxis_title="Update Sparsity",
        yaxis=dict(range=[0,1])  # since sparsity ∈ [0,1]
    )
    fig_global.write_html(os.path.join(out_dir, "global_sparsity.html"))

    # --- 2. Layer-wise sparsity (heatmap) ---
    # Collect all layer names
    all_layers = sorted({layer for ckpt in data for layer in data[ckpt]["layers"].keys()})
    z = []
    for ckpt in checkpoints:
        row = []
        for layer in all_layers:
            row.append(data[ckpt]["layers"].get(layer, None))
        z.append(row)

    fig_layers = px.imshow(
        z,
        x=all_layers,
        y=checkpoints,
        color_continuous_scale="Viridis",
        aspect="auto",
        labels=dict(x="Layers", y="Checkpoints", color="Update Sparsity")
    )
    fig_layers.update_layout(title="Layer-wise Update Sparsity Heatmap")
    fig_layers.write_html(os.path.join(out_dir, "layer_sparsity_heatmap.html"))

    # --- 3. Submodule sparsity (grouped bar chart) ---
    all_submodules = sorted({sub for ckpt in data for sub in data[ckpt]["submodules"].keys()})
    fig_sub = go.Figure()
    for ckpt in checkpoints:
        sub_vals = [data[ckpt]["submodules"].get(sub, None) for sub in all_submodules]
        fig_sub.add_trace(go.Bar(
            x=all_submodules,
            y=sub_vals,
            name=ckpt
        ))
    fig_sub.update_layout(
        title="Submodule Update Sparsity Across Checkpoints",
        barmode="group",
        xaxis_title="Submodules",
        yaxis_title="Update Sparsity"
    )
    fig_sub.write_html(os.path.join(out_dir, "submodule_sparsity.html"))

    print(f"✅ Plots saved to {out_dir}/")

def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    else:
        return obj


def get_name_of_run(checkpoint_path_1: str, checkpoint_path_2: str):
    return checkpoint_path_1.split('/')[-1] + "__" + checkpoint_path_2.split('/')[-1]
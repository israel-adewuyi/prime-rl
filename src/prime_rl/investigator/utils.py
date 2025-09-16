import plotly.graph_objects as go

def plot_weight_diffs(stats: dict, output_path: str = "weight_diffs.html", title: str = "Model Weight Differences") -> None:
    """
    Plot layer-wise differences for attention, MLP, and token embedding,
    and save as an interactive HTML file.

    Args:
        stats (dict): {
            "token_embedding": float,
            "attn": List[float],
            "mlp":  List[float]
        }
        output_path (str): Path to save the HTML file.
        title (str): Plot title.
    """
    # num_layers = len(stats["attn"])
    # layers = list(range(num_layers))

    fig = go.Figure()

    # # Attention bars
    # fig.add_trace(go.Bar(
    #     x=layers,
    #     y=stats["attn"],
    #     name="Attention Diff",
    #     marker_color="steelblue",
    #     hovertemplate="Layer %{x}<br>Attention Diff: %{y:.6f}<extra></extra>"
    # ))

    # # MLP bars
    # fig.add_trace(go.Bar(
    #     x=layers,
    #     y=stats["mlp"],
    #     name="MLP Diff",
    #     marker_color="seagreen",
    #     hovertemplate="Layer %{x}<br>MLP Diff: %{y:.6f}<extra></extra>"
    # ))

    # Token embedding (at x = -1)
    fig.add_trace(go.Bar(
        x=[-1],
        y=[stats["token_embedding"]],
        name="Token Embedding Diff",
        marker_color="darkorange",
        hovertemplate="Token Embedding<br>Diff: %{y:.6f}<extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Layer",
        yaxis_title="Mean Abs Diff",
        title=title,
        xaxis=dict(
            tickmode="linear",
            tick0=-1,
            dtick=1,
            title_standoff=10
        )
    )

    fig.write_html(output_path)

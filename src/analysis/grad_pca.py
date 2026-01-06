from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_gradient(step, threshold, local_dir="../../outputs", hf_repo_id=None, hf_token=None, force_redownload=False):
    """
    Load gradient from appropriate location with automatic repo existence check.
    """
    local_path = Path(local_dir)

    if step < threshold:
        # Load from local
        filename = local_path / f"0.5B_AS_grad_{step}.pt"
        if not filename.exists():
            raise FileNotFoundError(f"Local gradient {step} not found: {filename}")
        return torch.load(filename)
    else:
        # Load from HuggingFace
        if hf_repo_id is None:
            raise ValueError("HF repo ID must be provided for steps >= threshold")

        return load_from_hf(step, hf_repo_id, local_path, hf_token=hf_token, force_redownload=force_redownload)


import os
import time

from huggingface_hub import hf_hub_download


def load_from_hf(step, hf_repo_id, local_cache_dir, hf_token=None, force_redownload=False, max_retries=5):
    """
    Load gradient from HuggingFace Hub with robust retry logic.
    Compatible with older huggingface_hub versions.
    """
    print(f"Downloading gradient {step} from HuggingFace...")

    filename = f"gradients/0.5B_AS_grad_{step}.pt"

    # Set environment variables for better performance
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")

            # Remove 'timeout' parameter for compatibility
            downloaded_file = hf_hub_download(
                repo_id=hf_repo_id,
                filename=filename,
                repo_type="model",
                token=hf_token,
                # cache_dir=str(local_cache_dir) if local_cache_dir else None,
                force_download=force_redownload,
                # timeout=30.0,  # REMOVED - not available in your version
            )

            grad = torch.load(downloaded_file)
            print(f"✓ Successfully downloaded gradient {step}")
            return grad

        except Exception as e:
            error_msg = str(e)

            # Check for timeout/504 errors
            if any(keyword in error_msg for keyword in ["504", "Gateway Timeout", "timeout", "Timeout"]):
                print(f"⚠️  Timeout error (attempt {attempt + 1}/{max_retries})")
            elif "Connection" in error_msg:
                print(f"⚠️  Connection error (attempt {attempt + 1}/{max_retries})")
            else:
                print(f"⚠️  Error: {error_msg} (attempt {attempt + 1}/{max_retries})")

            # If this is the last attempt, raise the error
            if attempt == max_retries - 1:
                print(f"❌ Failed to download gradient {step} after {max_retries} attempts")
                raise FileNotFoundError(f"Gradient {step} not found after {max_retries} attempts")

            # Exponential backoff
            wait_time = 2 ** (attempt + 1)
            print(f"⏳ Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)


N = 150
device = "cuda"

# Pre-allocate result matrix on CPU (small: 10×10)
GGT = torch.zeros((N, N), dtype=torch.float32)

# Load gradients incrementally using symmetry
for i in range(N):
    # Load gradient i once and keep it
    print(f"Loading gradient {i}...")
    grad_i = load_gradient(
        step=i, threshold=75, hf_repo_id="israel-adewuyi/gradients", hf_token="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"
    ).to(device)
    # grad_i = torch.load(f"../../outputs/0.5B_AS_grad_{i}.pt").to(device)

    # Compute diagonal element
    GGT[i, i] = torch.dot(grad_i, grad_i).cpu()

    # Only compute upper triangular part
    for j in range(i + 1, N):
        grad_j = load_gradient(
            step=j,
            threshold=75,
            hf_repo_id="israel-adewuyi/gradients",
            hf_token="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",
        ).to(device)
        val = torch.dot(grad_i, grad_j).cpu()
        GGT[i, j] = val
        GGT[j, i] = val  # Symmetric
        del grad_j  # Free GPU memory

    del grad_i  # Free GPU memory
    torch.cuda.empty_cache()  # Optional: clear cache


def plot_and_save_eigenspectrum(eigenvalues, save_dir="./plots", filename_prefix="eigenspectrum"):
    """
    Plot eigenvalues and save figures to files.

    Args:
        eigenvalues: Tensor or numpy array of eigenvalues (sorted descending)
        save_dir: Directory to save plots
        filename_prefix: Prefix for saved files
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    eigenvalues_np = eigenvalues.numpy() if torch.is_tensor(eigenvalues) else eigenvalues

    # Calculate statistics
    total_variance = eigenvalues_np.sum()
    cumulative_variance = np.cumsum(eigenvalues_np) / total_variance

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Regular eigenvalue plot
    axes[0].plot(eigenvalues_np, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Component Index", fontsize=11)
    axes[0].set_ylabel("Eigenvalue", fontsize=11)
    axes[0].set_title("Eigenvalue Spectrum", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="both", which="major", labelsize=10)

    # Add eigenvalue values on plot
    for i, val in enumerate(eigenvalues_np[:5]):  # Label first 5
        axes[0].text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # 2. Log-scale plot
    axes[1].semilogy(eigenvalues_np, "ro-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Component Index", fontsize=11)
    axes[1].set_ylabel("Eigenvalue (log scale)", fontsize=11)
    axes[1].set_title("Log-scale Eigenvalue Spectrum", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].tick_params(axis="both", which="major", labelsize=10)

    # 3. Cumulative explained variance
    axes[2].plot(cumulative_variance, "go-", linewidth=2, markersize=8, label="Cumulative Variance")
    axes[2].set_xlabel("Number of Components", fontsize=11)
    axes[2].set_ylabel("Cumulative Explained Variance", fontsize=11)
    axes[2].set_title("Cumulative Explained Variance", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.95, color="r", linestyle="--", alpha=0.7, linewidth=1.5, label="95% threshold")
    axes[2].axhline(y=0.99, color="b", linestyle="--", alpha=0.7, linewidth=1.5, label="99% threshold")
    axes[2].legend(fontsize=10)
    axes[2].tick_params(axis="both", which="major", labelsize=10)

    # Add percentage labels
    for i, val in enumerate(cumulative_variance):
        if i % 2 == 0 or i == len(cumulative_variance) - 1:  # Label every other or last
            axes[2].text(i, val, f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    # Add overall title
    fig.suptitle("Eigenspectrum Analysis for Rank Determination", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save the figure
    save_paths = []

    # Save as single combined figure
    combined_path = save_path / f"{filename_prefix}_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight", facecolor="white")
    save_paths.append(combined_path)
    print(f"✓ Saved combined plot to: {combined_path}")

    # Also save individual subplots
    titles = ["eigenvalue_spectrum", "log_spectrum", "cumulative_variance"]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        # Create individual figure for each subplot
        fig_ind, ax_ind = plt.subplots(figsize=(8, 6))

        # Copy the subplot content
        for line in ax.get_lines():
            xdata, ydata = line.get_data()
            ax_ind.plot(
                xdata,
                ydata,
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=2,
                marker="o" if "o" in line.get_marker() else None,
                markersize=8,
            )

        # Copy grid
        ax_ind.grid(True, alpha=0.3)

        # Copy labels and title
        ax_ind.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax_ind.set_ylabel(ax.get_ylabel(), fontsize=12)
        ax_ind.set_title(ax.get_title(), fontsize=13, fontweight="bold")
        ax_ind.tick_params(axis="both", which="major", labelsize=11)

        # Copy horizontal lines if they exist
        for line in ax.get_lines():
            if line.get_linestyle() == "--":  # This identifies threshold lines
                ax_ind.axhline(
                    y=line.get_ydata()[0],
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    label=line.get_label() if hasattr(line, "get_label") else None,
                )

        if title == "cumulative_variance":
            ax_ind.legend(fontsize=11)

        # Save individual plot
        ind_path = save_path / f"{filename_prefix}_{title}.png"
        fig_ind.savefig(ind_path, dpi=300, bbox_inches="tight", facecolor="white")
        save_paths.append(ind_path)
        print(f"  → Saved {title} to: {ind_path}")
        plt.close(fig_ind)

    # Also save as PDF (vector format, good for publications)
    pdf_path = save_path / f"{filename_prefix}_combined.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    save_paths.append(pdf_path)
    print(f"✓ Saved PDF to: {pdf_path}")

    plt.show()
    plt.close(fig)

    return save_paths, eigenvalues_np, cumulative_variance


torch.save(GGT, f"gram_matrix_0.5B_AlphabetSort_N={N}.pt")

# For symmetric matrix A
eigenvalues, eigenvectors = torch.linalg.eigh(GGT)  # Returns eigenvalues in ascending order
eigenvalues = torch.flip(eigenvalues, dims=[0])  # Sort descending

save_paths, eigenvalues_np, cumulative_variance = plot_and_save_eigenspectrum(
    eigenvalues, save_dir="./eigen_analysis_plots", filename_prefix="matrix_eigenspectrum"
)

total_energy = eigenvalues.sum()
print(f"Total energy in gradients: {total_energy:.4f}")

# Proportion of total energy
energy_proportion = eigenvalues / total_energy
print("\nEnergy proportion per component:")
for i, prop in enumerate(energy_proportion[:5]):
    print(f"  Component {i + 1}: {prop:.3%}")

# Cumulative
cumulative = torch.cumsum(energy_proportion, dim=0)
print("\nCumulative energy:")
for i, cum in enumerate(cumulative[:5]):
    print(f"  First {i + 1} components: {cum:.3%}")

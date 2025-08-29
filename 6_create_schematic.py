# FILE: paper_fig_schematic_publication.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Adjust these imports to match your project structure
from config import PRIOR_LOW, PRIOR_HIGH, TOTAL_DURATION, FIGURES_PATH, MAX_NUM_POINTS
from utils import (
    simulate_microlensing_event_pytorch,
    ensure_dir,
    generate_observation_times,
)

# --- Helper Functions for Plotting ---


def style_ax(ax, remove_ticks=False):
    """Applies a consistent, minimalist style to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.tick_params(axis="x", colors="grey", length=4)
    ax.tick_params(axis="y", colors="grey", length=4)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


# --- Main Figure Generation ---


def create_schematic_figure_publication():
    """
    Generates a final, publication-ready schematic of the SBI methodology.
    This version features a clean, minimalist design with no connecting arrows.
    """
    ensure_dir(FIGURES_PATH)

    # Create the figure in a landscape aspect ratio with 4 columns
    fig = plt.figure(figsize=(11, 3))
    # wspace increases horizontal spacing between panels
    gs = fig.add_gridspec(1, 4, wspace=0.4)

    # --- 1. Panel (a): Sample Parameters ---
    ax1 = fig.add_subplot(gs[0, 0])
    # ax1.set_title("(a) Sample Parameters", fontsize=11, pad=10)

    param_0_samples = np.random.uniform(0, 1.5, 500)
    param_1_samples = np.random.uniform(-1, 1, 500)
    ax1.scatter(
        param_0_samples, param_1_samples, s=2, alpha=0.5, color="C0", rasterized=False
    )
    ax1.set_xlabel(r"$\theta_0$", fontsize=14)
    ax1.set_ylabel(r"$\theta_1$", fontsize=14)
    style_ax(ax1, remove_ticks=True)  # Remove tick labels
    ax1.set_facecolor("#f0f0f0")

    # --- 2. Panel (b): Simulate and Augment ---
    ax2 = fig.add_subplot(gs[0, 1])
    # ax2.set_title("(b) Simulate & Augment", fontsize=11, pad=10)

    true_params = torch.tensor(
        [
            12.0,  # t₀: Centered event
            0.25,  # u₀: High magnification
            np.log10(4.0),  # tE: A moderately long event (4 days)
            np.log10(0.05),  # ρ: Clear finite-source effects
            0.8,  # blend_fs: Mostly source flux
        ],
        dtype=torch.float32,
    )

    # --- 2. Generate the "Ideal" Underlying Physics ---
    # A smooth, dense, noiseless light curve.
    times_ideal = np.linspace(0, TOTAL_DURATION, 1000)
    flux_ideal = (
        simulate_microlensing_event_pytorch(true_params, times_ideal, phot_err=0.0)
        .cpu()
        .numpy()
    )

    # --- 3. Generate one "Augmented" Observation ---
    # This mimics the data the network actually sees during training.
    # We fix the seed for reproducibility of the figure.
    np.random.seed(1)

    # Generate realistic observation times with gaps and dropout.
    # Let's use specific, illustrative augmentation parameters for this figure.
    times_augmented = generate_observation_times(
        num_gaps_range=(2, 2),  # seasonal gap
        gap_length_range=(2, 3),  # 4-5 days long
        dropout_rate_range=(0.5, 0.5),  # Moderate dropout
    )

    # Simulate the flux with photometric noise.
    phot_err_augmented = 0.02  # A typical noise level
    flux_augmented = (
        simulate_microlensing_event_pytorch(
            true_params, times_augmented, phot_err=phot_err_augmented
        )
        .cpu()
        .numpy()
    )

    # theta_example = torch.tensor([10.0, 0.3, np.log10(5.0), np.log10(0.05), 0.9])
    # times_ideal = np.linspace(0, TOTAL_DURATION, 500)
    # flux_ideal = simulate_microlensing_event_pytorch(theta_example, times_ideal).cpu().numpy()

    # np.random.seed(42)
    # keep_indices = np.random.choice(len(times_ideal), size=int(0.3 * len(times_ideal)), replace=False)
    # gap_mask = (times_ideal > 6) & (times_ideal < 11)
    # keep_indices = keep_indices[~gap_mask[keep_indices]]
    # times_aug = times_ideal[keep_indices]
    # flux_aug = flux_ideal[keep_indices] + np.random.normal(0, 0.01, size=len(keep_indices))

    ax2.plot(
        times_ideal,
        flux_ideal,
        color="grey",
        ls="--",
        lw=1,
        alpha=0.7,
        label="Ideal Model",
    )
    ax2.errorbar(
        times_augmented,
        flux_augmented,
        yerr=phot_err_augmented,
        fmt=".",
        color="C0",
        ms=4,
        label="Observation",
        rasterized=False,
    )
    ax2.set_xlabel("Time", fontsize=14)
    ax2.set_ylabel("Magnification", fontsize=14)  # Changed from "Flux"
    # ax2.legend(loc='upper right', fontsize='small', frameon=False)
    style_ax(ax2)
    ax2.set_ylim(bottom=0.95)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # --- 3. Panel (c): Process with Transformer ---
    ax3 = fig.add_subplot(gs[0, 2])
    # ax3.set_title("(c) Process with Transformer", fontsize=11, pad=10)
    style_ax(ax3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines["left"].set_visible(False)  # Remove axis lines for this abstract panel
    ax3.spines["bottom"].set_visible(False)

    # Redesigned, cleaner architecture diagram
    # Input sequence
    # for i in range(4):
    #     color = 'C0' if i < 3 else 'lightgrey'
    #     rect = patches.Rectangle((i*0.8, -0.4), 0.7, 0.8, facecolor=color, edgecolor='black', lw=0.5)
    #     ax3.add_patch(rect)
    # ax3.text(1.2, -1.2, "Input Sequence", ha='center', fontsize=9)

    # # Arrow symbol for transformation
    # ax3.text(3.85, 0, r"$\rightarrow$", ha='center', va='center', fontsize=20)

    # # Encoder block
    # encoder_box = patches.Rectangle((4.9, -0.5), 2, 1, facecolor='#f0f0f0', edgecolor='black', lw=1.5)
    # ax3.add_patch(encoder_box)
    # ax3.text(5.9, 0.8, "Transformer\nEncoder", ha='center', va='center', weight='bold', fontsize=8)

    # # Arrow symbol for transformation
    # ax3.text(7.75, 0, r"$\rightarrow$", ha='center', va='center', fontsize=20)

    # # Output vector
    # output_vec = patches.Rectangle((8.5, -0.4), 0.8, 0.8, facecolor='C2', edgecolor='black', lw=1)
    # ax3.add_patch(output_vec)
    # ax3.text(8.9, -1.2, "Summary\nVector", ha='center', fontsize=9)

    # --- 4. Panel (d): Infer Posterior ---
    ax4 = fig.add_subplot(gs[0, 3])
    # ax4.set_title("(d) Infer Posterior", fontsize=11, pad=10)

    from scipy.stats import multivariate_normal

    mean = [0, 0]
    cov = [[1.2, -0.8], [-0.8, 0.8]]
    x_grid, y_grid = np.mgrid[-2.5:2.5:0.01, -2:2:0.01]
    pos = np.dstack((x_grid, y_grid))
    rv = multivariate_normal(mean, cov)
    ax4.contourf(x_grid, y_grid, rv.pdf(pos), cmap="Blues", levels=5)
    ax4.set_xlabel(r"$\theta_0$", fontsize=14)
    ax4.set_ylabel(r"$\theta_1$", fontsize=14)
    style_ax(ax4, remove_ticks=True)  # Remove tick labels

    # --- Final Touches ---
    fig.tight_layout(pad=0.5, w_pad=1.5)

    output_path = os.path.join(FIGURES_PATH, "paper_fig_schematic_publication.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✓ Saved final schematic figure to: {output_path}")


if __name__ == "__main__":
    create_schematic_figure_publication()

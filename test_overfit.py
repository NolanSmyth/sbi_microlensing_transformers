# test_overfit.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import corner

# Import sbi components
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# Import your project modules
from config import *
from utils import generate_base_simulation  # We use this to get clean data
from model import TransformerEmbeddingNet


def main():
    """
    Performs an overfitting test to verify model capacity and training loop.
    The model should be able to perfectly memorize a single, clean batch of data.
    """
    print("--- Running Overfitting Sanity Check ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Create one single, clean batch of data ---
    # We will use this exact batch for all training epochs.
    overfit_batch_size = 32
    print(f"Generating one clean batch of {overfit_batch_size} simulations...")

    # Sample parameters from the prior
    thetas_overfit = PRIOR.sample((overfit_batch_size,))

    # Generate noiseless, full-cadence light curves using your utility.
    # Note: We need to pad these simulations just like in the real training.
    xs_unpadded = torch.stack(
        [generate_base_simulation(theta) for theta in thetas_overfit]
    )

    # Manually pad to MAX_NUM_POINTS
    num_base_points = xs_unpadded.shape[1]

    # If the base sim is longer than the model's max length, downsample uniformly.
    if num_base_points > MAX_NUM_POINTS:
        idx = (
            torch.linspace(
                0, num_base_points - 1, MAX_NUM_POINTS, device=xs_unpadded.device
            )
            .round()
            .long()
        )
        xs_unpadded = xs_unpadded.index_select(1, idx)

    # Now pad to MAX_NUM_POINTS (or just copy if already equal)
    seq_len = xs_unpadded.shape[1]
    xs_overfit = torch.full(
        (overfit_batch_size, MAX_NUM_POINTS, 2), fill_value=-2.0, device=device
    )
    xs_overfit[:, :, 1] = 0.0
    xs_overfit[:, :seq_len, :] = xs_unpadded.to(device)

    print(f"Shape of overfitting thetas: {thetas_overfit.shape}")
    print(f"Shape of overfitting xs: {xs_overfit.shape}")

    # --- 2. Instantiate a fresh, untrained model ---
    print("\nInstantiating a new, untrained model...")
    embedding_net = TransformerEmbeddingNet(
        d_model=D_MODEL, nhead=N_HEAD, d_hid=D_HID, nlayers=N_LAYERS, dropout=DROPOUT
    ).to(device)

    posterior_estimator_build_fn = posterior_nn(
        model="maf", embedding_net=embedding_net
    )
    inference = NPE(
        prior=PRIOR, density_estimator=posterior_estimator_build_fn, device=device
    )

    # Append the single batch of simulations
    inference.append_simulations(thetas_overfit, xs_overfit)

    # --- 3. Train repeatedly on this single batch ---
    print("Training for 1000 epochs on the *same* batch to force overfitting...")
    # We don't need validation or early stopping for this test.
    density_estimator = inference.train(
        training_batch_size=overfit_batch_size,  # Use the full batch
        max_num_epochs=1000,
        stop_after_epochs=1000,
        learning_rate=LEARNING_RATE,
        show_train_summary=True,
        validation_fraction=0.1,  # small validation set
    )

    # The final training loss should be very, very low (e.g., negative).
    final_loss = inference.summary["training_loss"][-1]
    print(f"\nFinal training loss: {final_loss:.4f}")
    assert (
        final_loss < -5.0
    ), "Overfitting Test FAILED: Final loss is not low enough. Model may not be learning."
    print(
        "Overfitting Test PASSED: Model successfully minimized loss on a single batch."
    )

    # --- 4. Evaluate parameter recovery on the overfitted data ---
    print("\nVerifying parameter recovery on a sample from the overfit batch...")

    posterior = inference.build_posterior(density_estimator)

    # Pick the first light curve from the batch to test
    test_theta = thetas_overfit[0]
    test_x = xs_overfit[0]

    # Sample from the posterior. It should be EXTREMELY sharp around the true value.
    posterior_samples = posterior.sample((5000,), x=test_x, show_progress_bars=True)

    # Create a corner plot. The posterior should be a tiny dot on the truth.
    fig = corner.corner(
        posterior_samples.cpu().numpy(),
        labels=PARAM_NAMES,
        truths=test_theta.cpu().numpy(),
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    fig.suptitle("Overfitting Test: Posterior Recovery", fontsize=16, y=1.02)
    plt.savefig(FIGURES_PATH + "/overfitting_posterior_corner_plot.png")
    plt.show()

    print(
        "\nTest complete. Check the corner plot. The posterior should be a delta function at the truth."
    )


if __name__ == "__main__":
    main()

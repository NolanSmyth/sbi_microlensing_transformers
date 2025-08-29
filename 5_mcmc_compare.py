# 5_mcmc_compare.py
import torch
import numpy as np
import corner
import matplotlib.pyplot as plt
from tqdm import tqdm
import emcee
import time

from config import *
from utils import *
from model import *

NUM_MCMC_SAMPLES = 18000
BURN_IN = 2000
# BURN_IN = 100
# NUM_MCMC_SAMPLES = 200
NUM_NPE_SAMPLES = 20000
N_WALKERS = 32


# --- Helper function for this script ---
def prepare_data_for_network(times, flux, device, errors=None):
    """Pads and formats a lightcurve to be fed into the transformer network."""
    num_obs = len(times)
    normalized_times = torch.tensor(
        times / (TOTAL_DURATION / 2.0) - 1.0, dtype=torch.float32
    )
    sigma_vec = torch.full((len(times),), float(errors), dtype=torch.float32)
    data = torch.stack(
        (normalized_times, flux.to(dtype=torch.float32, device="cpu"), sigma_vec), dim=1
    )

    padded_data = torch.full((MAX_NUM_POINTS, 3), fill_value=-2.0)
    padded_data[:, 1] = 0.0
    padded_data[:, 2] = 0.0

    padded_data[:num_obs, :] = data

    return padded_data.to(device)


def log_prior(params):
    """
    Log prior probability for MCMC sampling.
    params: [t_0, u_0, log10_t_E, log10_rho, blend_fs]
    """
    t_0, u_0, log10_t_E, log10_rho, blend_fs = params

    # Check bounds (same as in config.py)
    if not (0.0 <= t_0 <= TOTAL_DURATION):
        return -np.inf
    if not (0.0 <= u_0 <= 1.5):
        return -np.inf
    if not (np.log10(0.1) <= log10_t_E <= np.log10(20.0)):
        return -np.inf
    if not (np.log10(0.01) <= log10_rho <= np.log10(10.0)):
        return -np.inf
    if not (0.1 <= blend_fs <= 1.0):
        return -np.inf

    # Uniform prior within bounds
    return 0.0


def log_likelihood(params, times, flux, flux_err):
    """
    Log likelihood function for MCMC sampling.
    """
    try:
        # Convert to torch tensors for simulation
        params_tensor = torch.tensor(params, dtype=torch.float32)

        # Simulate the model lightcurve
        model_flux = simulate_microlensing_event_pytorch(
            params_tensor, times, phot_err=0.0
        )

        # Calculate chi-squared
        residuals = (flux.cpu().numpy() - model_flux.cpu().numpy()) / flux_err
        chi2 = np.sum(residuals**2)

        # Return log likelihood
        return -0.5 * chi2

    except Exception as e:
        # Return very low likelihood if simulation fails
        return -np.inf


def log_probability(params, times, flux, flux_err):
    """
    Log posterior probability (prior + likelihood).
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, times, flux, flux_err)


def check_mcmc_convergence(sampler, min_samples=100, tolerance=0.01, quiet=False):
    """
    Check MCMC convergence using autocorrelation time.

    Parameters:
    - sampler: emcee sampler object
    - min_samples: minimum number of samples before checking convergence
    - tolerance: convergence criterion (chain length / autocorr time > 1/tolerance)
    - quiet: suppress print statements

    Returns:
    - converged: boolean indicating if chains have converged
    - tau: autocorrelation times for each parameter
    - effective_samples: effective number of samples for each parameter
    """
    try:
        # Get current chain length
        # chain_length = sampler.get_chain().shape[0]
        chain_length, nwalkers, _ = sampler.get_chain().shape

        if chain_length < min_samples:
            if not quiet:
                print(
                    f"Not enough samples for convergence check (need {min_samples}, have {chain_length})"
                )
            return False, None, None

        # Calculate autocorrelation time
        try:
            tau = sampler.get_autocorr_time(
                tol=0
            )  # tol=0 to avoid warnings for short chains
        except emcee.autocorr.AutocorrError:
            if not quiet:
                print(
                    "Warning: Autocorrelation time estimation failed (chains too short)"
                )
            return False, None, None

        # Check convergence criterion: chain should be longer than ~50*tau for each parameter
        converged = np.all(chain_length > (1.0 / tolerance) * tau)

        # Calculate effective number of samples
        effective_samples = (
            nwalkers * chain_length / (2.0 * tau)
        )  # Factor of 2 for conservative estimate

        if not quiet:
            print(f"\n=== MCMC Convergence Check ===")
            print(f"Chain length: {chain_length}")
            print(f"Autocorrelation times: {tau}")
            print(f"Convergence criterion: chain_length > {1.0/tolerance:.0f} * tau")
            print(f"Effective samples per parameter: {effective_samples}")
            print(f"Converged: {converged}")

            # Individual parameter breakdown
            for i, param_name in enumerate(PARAM_NAMES):
                ratio = chain_length / tau[i]
                status = "✓" if ratio > (1.0 / tolerance) else "✗"
                print(
                    f"  {param_name:>15}: tau={tau[i]:6.1f}, ratio={ratio:6.1f}, eff_samples={effective_samples[i]:6.1f} {status}"
                )

        return converged, tau, effective_samples

    except Exception as e:
        if not quiet:
            print(f"Error in convergence check: {e}")
        return False, None, None


def run_mcmc_sampling(
    times, flux, flux_err, nwalkers=32, nsteps=5000, burn_in=1000, initial_params=None
):
    """
    Run MCMC sampling using emcee.
    """
    print(f"Running MCMC with {nwalkers} walkers for {nsteps} steps...")

    # Initialize walkers near a reasonable starting point
    # Use the same true parameters from 4_evaluate_model.py as a starting point
    # initial_params = np.array([8.67, 0.85, np.log10(0.8), np.log10(2.5), 0.62])
    # initial_params = np.array([8.67, 0.45, np.log10(0.8), np.log10(1.2), 0.62])

    # Add small random perturbations to create starting positions for all walkers
    # ndim = len(initial_params)
    if initial_params is None:
        # Sample from prior and convert to numpy
        pos = PRIOR.sample((nwalkers,)).cpu().numpy()
        print(f"Sampled initial positions shape: {pos.shape}")
    else:
        # Use provided initial parameters
        if initial_params.ndim == 1:
            # Single set of parameters - create variations for all walkers
            # ndim = len(initial_params)
            pos = initial_params + 0.01 * np.random.randn(nwalkers, ndim)
        else:
            # Already have positions for all walkers
            pos = initial_params
        print(f"Using provided initial positions shape: {pos.shape}")

    # Ensure all starting positions are within bounds
    for i in range(nwalkers):
        pos[i] = np.clip(
            pos[i],
            [0.0, 0.0, np.log10(0.1), np.log10(0.01), 0.1],
            [TOTAL_DURATION, 1.5, np.log10(20.0), np.log10(10.0), 1.0],
        )

    ndim = pos.shape[1]

    # Set up the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(times, flux, flux_err),
        # moves=WalkMove(),
    )

    # Run burn-in
    print("Running burn-in...")
    start_time = time.time()
    state = sampler.run_mcmc(pos, burn_in, progress=True)
    burn_time = time.time() - start_time
    sampler.reset()

    # Run production
    print("Running production...")
    production_start = time.time()

    # Run production with periodic convergence checks
    check_interval = min(500, nsteps // 10)  # Check every 500 steps or 10% of total
    for i in range(0, nsteps, check_interval):
        remaining_steps = min(check_interval, nsteps - i)
        sampler.run_mcmc(state, remaining_steps, progress=True)
        state = sampler.get_last_sample()

        # Check convergence periodically (but don't stop early for simplicity)
        if (
            i > 0 and (i + remaining_steps) >= nsteps // 2
        ):  # Only check after halfway point
            converged, tau, eff_samples = check_mcmc_convergence(sampler, quiet=True)
            if converged:
                print(f"✓ Chains converged after {i + remaining_steps} steps")
            else:
                print("✗ not yet converged. eff_samples: ", eff_samples)

    production_time = time.time() - production_start

    total_time = burn_time + production_time

    print(f"MCMC completed in {total_time:.2f} seconds")
    print(f"  Burn-in: {burn_time:.2f} seconds")
    print(f"  Production: {production_time:.2f} seconds")

    # Final convergence check
    print("\n" + "=" * 50)
    converged, tau, eff_samples = check_mcmc_convergence(sampler, tolerance=0.01)
    print("=" * 50)

    # Get samples
    samples = sampler.get_chain(discard=0, flat=True)
    print(len(samples), "samples obtained")

    # Print acceptance fraction
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    # Recommendations based on convergence
    if converged:
        print("✓ MCMC chains have converged successfully!")
    else:
        print("⚠  MCMC chains may not have converged. Consider:")
        print("   - Increasing the number of steps")
        print("   - Checking parameter initialization")
        print("   - Examining chain traces")

    return samples, total_time, sampler


def create_lightcurve_comparison_plot(
    npe_samples,
    mcmc_samples,
    true_params,
    observation_times,
    flux,
    noise_level,
    save_path,
):
    """
    Create a lightcurve plot showing data, truth, and recovered models with percentile bands.
    """
    print("Creating lightcurve comparison plot with percentile bands...")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Generate smooth times for model curves
    smooth_times = np.linspace(0, TOTAL_DURATION, 1000)

    # True lightcurve
    true_lc_smooth = simulate_microlensing_event_pytorch(
        torch.tensor(true_params), smooth_times, phot_err=0.0
    )

    # Generate lightcurves for percentile calculation
    print("  Computing NPE lightcurve percentiles...")
    npe_lightcurves = []
    npe_subset = npe_samples[
        :: max(1, len(npe_samples) // 200)
    ]  # Use every nth sample, max 200
    for params in tqdm(npe_subset, desc="NPE lightcurves"):
        lc = simulate_microlensing_event_pytorch(
            torch.tensor(params), smooth_times, phot_err=0.0
        )
        npe_lightcurves.append(lc.cpu().numpy())
    npe_lightcurves = np.array(npe_lightcurves)

    print("  Computing MCMC lightcurve percentiles...")
    mcmc_lightcurves = []
    mcmc_subset = mcmc_samples[
        :: max(1, len(mcmc_samples) // 200)
    ]  # Use every nth sample, max 200
    for params in tqdm(mcmc_subset, desc="MCMC lightcurves"):
        lc = simulate_microlensing_event_pytorch(
            torch.tensor(params), smooth_times, phot_err=0.0
        )
        mcmc_lightcurves.append(lc.cpu().numpy())
    mcmc_lightcurves = np.array(mcmc_lightcurves)

    # Calculate percentiles
    npe_median = np.percentile(npe_lightcurves, 50, axis=0)
    npe_lower = np.percentile(npe_lightcurves, 16, axis=0)
    npe_upper = np.percentile(npe_lightcurves, 84, axis=0)

    mcmc_median = np.percentile(mcmc_lightcurves, 50, axis=0)
    mcmc_lower = np.percentile(mcmc_lightcurves, 16, axis=0)
    mcmc_upper = np.percentile(mcmc_lightcurves, 84, axis=0)

    # Plot data with error bars
    ax.errorbar(
        observation_times,
        flux.cpu(),
        yerr=noise_level,
        fmt="o",
        color="black",
        alpha=0.7,
        capsize=4,
        markersize=5,
        label="Data",
        zorder=0,
    )

    # Plot true lightcurve
    # ax.plot(smooth_times, true_lc_smooth.cpu(),
    #         color='dodgerblue', linewidth=3, label='True Lightcurve', zorder=1)

    # Plot NPE percentile bands
    ax.fill_between(smooth_times, npe_lower, npe_upper, color="#0072B2", alpha=0.3)
    ax.plot(
        smooth_times, npe_median, color="#0072B2", linewidth=3, label="NPE", zorder=2
    )

    # Plot MCMC percentile bands
    ax.fill_between(smooth_times, mcmc_lower, mcmc_upper, color="#D55E00", alpha=0.3)
    ax.plot(
        smooth_times,
        mcmc_median,
        color="#D55E00",
        linewidth=3,
        label="MCMC",
        zorder=2,
        ls="--",
    )

    ax.set_xlabel("Time (days)", fontsize=16)
    ax.set_ylabel("Relative Flux", fontsize=16)
    # ax.set_title(f'Lightcurve Comparison: Data vs Recovered Models (Noise = {noise_level:.3f})',
    #  fontsize=16)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print(f"Lightcurve comparison plot saved to: {save_path}")


def create_corner_overlay_plot(
    npe_samples, mcmc_samples, true_params, noise_level, save_path
):
    """
    Create a corner plot with overlaid contours from both methods.
    """
    print("Creating corner overlay plot...")

    ranges = [
        (0.0, TOTAL_DURATION),  # t0
        (0.0, 1.5),  # u0
        (np.log10(0.1), np.log10(20.0)),  # log10(tE)
        (np.log10(0.01), np.log10(10.0)),  # log10(rho)
        (0.1, 1.0),  # fs
    ]

    # Create the base corner plot with NPE samples
    mins = np.minimum(npe_samples.min(axis=0), mcmc_samples.min(axis=0))
    maxs = np.maximum(npe_samples.max(axis=0), mcmc_samples.max(axis=0))
    ranges_auto = [
        (max(a, lo), min(b, hi)) for (a, b), (lo, hi) in zip(zip(mins, maxs), ranges)
    ]

    # Overlay MCMC contours
    fig = corner.corner(
        mcmc_samples,
        range=ranges_auto,
        # fig=fig,
        labels=PARAM_NAMES,
        color="#D55E00",
        truths=true_params,
        truth_color="#009E73",
        show_titles=False,
        title_kwargs={"fontsize": 18},
        label_kwargs={"fontsize": 16},
        hist_kwargs={"alpha": 0.6, "density": True},
        contour_kwargs={"alpha": 0.8},
        # quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.864],
        plot_datapoints=False,
        smooth=True,
        fill_contours=True,
    )

    corner.corner(
        npe_samples,
        range=ranges_auto,
        fig=fig,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.864],  # 2d 1 and 2 sigma
        color="#0072B2",
        hist_kwargs={"alpha": 0.6, "density": True},
        contour_kwargs={"alpha": 0.6},
        plot_datapoints=False,
        smooth=True,
        fill_contours=True,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print(f"Corner overlay plot saved to: {save_path}")


def create_trace_plots(sampler, true_params, burn_in, save_path):
    """
    Create trace plots to visualize chain behavior and convergence.
    """
    print("Creating trace plots...")

    # Get the chain (with burn-in included for visualization)
    chain = sampler.get_chain()
    nsteps, nwalkers, ndim = chain.shape

    # Create figure
    fig, axes = plt.subplots(ndim, 1, figsize=(12, 2 * ndim))
    if ndim == 1:
        axes = [axes]

    for i in range(ndim):
        ax = axes[i]

        # Plot all walker chains
        for walker in range(min(nwalkers, 20)):  # Limit to 20 walkers for clarity
            ax.plot(chain[:, walker, i], alpha=0.3, color="gray", linewidth=0.5)

        # Plot the median chain in a different color
        median_chain = np.median(chain[:, :, i], axis=1)
        ax.plot(median_chain, color="blue", linewidth=2, alpha=0.8, label="Median")

        # Mark burn-in period
        # if burn_in > 0:
        #     ax.axvline(burn_in, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Burn-in end')

        # Mark true value
        ax.axhline(
            true_params[i],
            color="green",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label="True value",
        )

        ax.set_ylabel(PARAM_NAMES[i], fontsize=12)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=10)
        if i == ndim - 1:
            ax.set_xlabel("Step", fontsize=12)

    plt.suptitle("MCMC Chain Traces", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Trace plots saved to: {save_path}")


def main():
    ensure_dir(FIGURES_PATH)
    print(f"Loading trained posterior from {LOADED_POSTERIOR_PATH}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    saved = load_saved_cpu(LOADED_POSTERIOR_PATH)
    prior, posterior = build_posterior_from_saved(saved, device=device)

    print("--- MCMC vs NPE Comparison ---")

    # Use the same setup as in 4_evaluate_model.py
    np.random.seed(0)  # for reproducibility
    # true_params_test = torch.tensor([8.67, 0.85, np.log10(0.8), np.log10(2.5), 0.62])
    true_params_test = torch.tensor([8.67, 0.45, np.log10(0.6), np.log10(0.8), 0.62])
    # true_params_test = torch.tensor(PRIOR.sample((1,)).squeeze().cpu())

    print(f"True parameters: {true_params_test}")

    # Generate observation times (same as in evaluation script)
    observation_times = generate_observation_times(
        num_gaps_range=(2, 2), gap_length_range=(1, 1.5), dropout_rate_range=(0.4, 0.4)
    )

    # observation_times = generate_observation_times(num_gaps_range=(0, 0),
    #                                              gap_length_range=(1, 1.5),
    #                                              dropout_rate_range=(0.6, 0.6))

    # Test different noise levels
    # noise_levels = [0.001, 0.01, 0.02]
    noise_levels = [0.015]
    results = {}

    for noise in noise_levels:
        print(f"\n=== Processing noise level: {noise} ===")

        # Simulate data
        flux = simulate_microlensing_event_pytorch(
            true_params_test, observation_times, phot_err=noise
        )

        # --- NPE Inference ---
        print("Running NPE inference...")
        data_net = prepare_data_for_network(
            observation_times, flux, device, errors=noise
        )

        npe_start = time.time()
        npe_samples = posterior.sample(
            (NUM_NPE_SAMPLES,), x=data_net, show_progress_bars=True
        )
        npe_time = time.time() - npe_start

        print(f"NPE completed in {npe_time:.2f} seconds")

        # --- MCMC Inference ---
        print("Running MCMC inference...")
        initial_params = true_params_test * (
            1 + 0.01 * np.random.randn(N_WALKERS, len(true_params_test))
        )

        mcmc_samples, mcmc_time, sampler = run_mcmc_sampling(
            observation_times,
            flux,
            noise,
            nwalkers=N_WALKERS,
            nsteps=NUM_MCMC_SAMPLES,
            burn_in=BURN_IN,
            initial_params=initial_params,
        )

        # Store results
        results[noise] = {
            "true_params": true_params_test.cpu().numpy(),
            "observation_times": observation_times,
            "flux": flux,
            "npe_samples": npe_samples.cpu().numpy(),
            "mcmc_samples": mcmc_samples,
            "npe_time": npe_time,
            "mcmc_time": mcmc_time,
            "sampler": sampler,
        }

        # --- Create overlay plots ---
        print("Creating overlay plots...")

        # 1. Lightcurve comparison plot
        lightcurve_path = f"{FIGURES_PATH}/lightcurve_comparison_noise_{noise:.3f}.pdf"
        create_lightcurve_comparison_plot(
            npe_samples.cpu().numpy(),
            mcmc_samples,
            true_params_test.cpu().numpy(),
            observation_times,
            flux,
            noise,
            lightcurve_path,
        )

        # 2. Corner plot overlay
        corner_overlay_path = f"{FIGURES_PATH}/corner_overlay_noise_{noise:.3f}.pdf"
        create_corner_overlay_plot(
            npe_samples.cpu().numpy(),
            mcmc_samples,
            true_params_test.cpu().numpy(),
            noise,
            corner_overlay_path,
        )

        print(f"Results for noise {noise}:")
        print(f"  NPE time: {npe_time:.2f} seconds")
        print(f"  MCMC time: {mcmc_time:.2f} seconds")
        print(f"  Speedup factor: {mcmc_time/npe_time:.1f}x")

    # --- Summary comparison ---
    print("\n=== SUMMARY ===")
    print("Noise Level | NPE Time (s) | MCMC Time (s) | Speedup Factor")
    print("-" * 60)
    for noise in noise_levels:
        npe_t = results[noise]["npe_time"]
        mcmc_t = results[noise]["mcmc_time"]
        speedup = mcmc_t / npe_t
        print(f"{noise:>10.3f} | {npe_t:>11.2f} | {mcmc_t:>12.2f} | {speedup:>13.1f}x")

    trace_figure_name = f"mcmc_traces_noise_{noise:.3f}.pdf"
    create_trace_plots(
        sampler,
        true_params_test.cpu().numpy(),
        BURN_IN,
        f"{FIGURES_PATH}/{trace_figure_name}",
    )

    print(f"\nAll results saved to {FIGURES_PATH}/")


if __name__ == "__main__":
    main()

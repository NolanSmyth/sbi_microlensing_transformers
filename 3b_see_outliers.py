# 3b_see_outliers.py
import os, dill
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from config import *
from model import TransformerEmbeddingNet
from utils import generate_augmented_observation, simulate_microlensing_event_pytorch, load_saved_cpu, build_posterior_from_saved, compute_peak_snr, compute_peak_snr_exact, check_recoverable

# ---------------------------- Controls ----------------------------------------- #
NUM_DIAG_SIMS        = 400           # injected cases to try
NUM_POST_SAMPLES     = 5000          # posterior samples per kept case
SHOW_PROGRESS_BARS   = False

# Recoverability rule: SNR-based with minimum point count
MIN_POINTS_IN_WINDOW = 10
MIN_PEAK_SNR = 5.0  # Minimum signal-to-noise ratio for recoverability

# Outlier analysis settings
NUM_WORST_CASES      = 12            # Number of worst cases to visualize
OUTLIER_METRIC       = "mse"         # "mse", "mad", or "chi2"

# ---------------------------- Helpers ------------------------------------------ #
def iqr_widths(samples_np):
    q16 = np.quantile(samples_np, 0.16, axis=0)
    q84 = np.quantile(samples_np, 0.84, axis=0)
    return (q84 - q16), q16, q84

def get_event_shape(posterior):
    for attr in ("_x_event_shape", "_event_shape_x", "_x_shape", "x_event_shape"):
        s = getattr(posterior, attr, None)
        if s is not None:
            try:
                t = tuple(s)
                if len(t) > 0:
                    return t
            except Exception:
                pass
    return None

def coerce_x_to_event_shape(x: torch.Tensor, event_shape) -> torch.Tensor:
    if x.dim() == len(event_shape):
        x = x.unsqueeze(0)
    if tuple(x.shape[-len(event_shape):]) == tuple(event_shape):
        return x
    if x.dim() >= 2:
        xt = x.transpose(-2, -1).contiguous()
        if tuple(xt.shape[-len(event_shape):]) == tuple(event_shape):
            return xt
    raise AssertionError(
        f"x trailing shape {tuple(x.shape[-len(event_shape):])} does not match "
        f"expected event_shape {tuple(event_shape)}; x full shape={tuple(x.shape)}"
    )


def count_points_in_tE_window(x_event: torch.Tensor, theta: torch.Tensor) -> int:
    """
    x_event: [MAX_NUM_POINTS, 2] with time channel normalized to [-1,1],
             padding time set to -2.0 (per your utils).
    theta:   [D] with order [t0, u0, log10_tE, log10_rho, blend_fs]
    """
    norm_t = x_event[:, 0]                         # normalized times
    valid = norm_t > -1.5                          # exclude padding (-2)
    # map back to simulation-time coordinates [0, TOTAL_DURATION]
    times = (norm_t[valid] + 1.0) * (TOTAL_DURATION / 2.0)

    t0  = float(theta[0].cpu())
    tE  = float(10.0 ** float(theta[2].cpu()))
    lo, hi = t0 - tE/2, t0 + tE/2

    return int(((times >= lo) & (times <= hi)).sum().item())


def compute_outlier_score(true_theta, recovered_theta, posterior_samples, metric="mse"):
    """
    Compute a score indicating how badly the recovery failed.
    Higher scores = worse performance.
    """
    if metric == "mse":
        # Mean squared error between true and recovered (median)
        return np.mean((true_theta - recovered_theta) ** 2)
    
    elif metric == "mad":
        # Mean absolute deviation
        return np.mean(np.abs(true_theta - recovered_theta))
    
    elif metric == "chi2":
        # Chi-squared like metric using posterior std
        posterior_std = np.std(posterior_samples, axis=0)
        # Avoid division by zero
        posterior_std = np.maximum(posterior_std, 1e-6)
        chi2 = np.sum(((true_theta - recovered_theta) / posterior_std) ** 2)
        return chi2
    
    else:
        raise ValueError(f"Unknown metric: {metric}")

def plot_light_curve_and_residuals(x_event, theta_true, theta_recovered, ax_lc, ax_res, case_idx):
    """
    Plot the observed light curve and residuals for a given case.
    """
    # Extract valid observations (non-padded)
    norm_times = x_event[:, 0].cpu().numpy()
    mags = x_event[:, 1].cpu().numpy()
    valid_mask = norm_times > -1.5  # exclude padding
    
    times_valid = norm_times[valid_mask]
    mags_valid = mags[valid_mask]
    
    # Convert normalized times back to original scale
    times_original = (times_valid + 1.0) * (TOTAL_DURATION / 2.0)
    
    # Generate model predictions for both true and recovered parameters
    # Create dense time grid for smooth model curves
    t_dense = np.linspace(0, TOTAL_DURATION, 1000)
    
    # Generate model light curves (you may need to adapt this based on your forward_model function)
    try:
        # True model
        theta_true_np = theta_true.cpu().numpy() if torch.is_tensor(theta_true) else theta_true
        mag_true = simulate_microlensing_event_pytorch(torch.tensor(theta_true_np), torch.tensor(t_dense)).numpy()
        
        # Recovered model  
        theta_rec_np = theta_recovered if not torch.is_tensor(theta_recovered) else theta_recovered.cpu().numpy()
        mag_recovered = simulate_microlensing_event_pytorch(torch.tensor(theta_rec_np), torch.tensor(t_dense)).numpy()
        
    except Exception as e:
        print(f"Warning: Could not generate model curves for case {case_idx}: {e}")
        mag_true = np.full_like(t_dense, np.nan)
        mag_recovered = np.full_like(t_dense, np.nan)
    
    # Plot light curve
    ax_lc.scatter(times_original, mags_valid, c='black', s=10, alpha=0.7, label='Observed')
    ax_lc.plot(t_dense, mag_true, 'r-', linewidth=2, label='True model')
    ax_lc.plot(t_dense, mag_recovered, 'b--', linewidth=2, label='Recovered model')
    ax_lc.set_ylabel('Magnitude')
    ax_lc.set_title(f'Case {case_idx}: Light Curve')
    ax_lc.legend(fontsize=8)
    ax_lc.grid(True, alpha=0.3)
    
    # Plot residuals (observed - true model)
    if not np.all(np.isnan(mag_true)):
        # Interpolate true model to observation times
        mag_true_interp = np.interp(times_original, t_dense, mag_true)
        residuals = mags_valid - mag_true_interp
        ax_res.scatter(times_original, residuals, c='red', s=10, alpha=0.7)
        ax_res.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax_res.set_ylabel('Residuals')
        ax_res.set_xlabel('Time')
        ax_res.grid(True, alpha=0.3)
    else:
        ax_res.text(0.5, 0.5, 'Model unavailable', transform=ax_res.transAxes, 
                   ha='center', va='center')

def format_parameter_text(theta, param_names):
    """Format parameter values for display."""
    text_lines = []
    for i, (val, name) in enumerate(zip(theta, param_names)):
        if 'log10' in name:
            # Show both log and linear values for log parameters
            linear_val = 10**val
            text_lines.append(f'{name}: {val:.3f} ({linear_val:.2e})')
        else:
            text_lines.append(f'{name}: {val:.3f}')
    return '\n'.join(text_lines)

# ---------------------------- Main --------------------------------------------- #
def main():
    os.makedirs(FIGURES_PATH, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    ckpt_path = globals().get("LOADED_POSTERIOR_PATH", None)
    if ckpt_path is None:
        raise RuntimeError("LOADED_POSTERIOR_PATH not found in config.")

    print(f"Loading posterior from {ckpt_path} …")
    saved = load_saved_cpu(ckpt_path)
    prior, posterior = build_posterior_from_saved(saved, device=device)

    # --- Injected parameters ---------------------------------------------------------
    print(f"Sampling {NUM_DIAG_SIMS} injected parameter vectors from prior …")
    thetas = prior.sample((NUM_DIAG_SIMS,))
    theta_np = thetas.cpu().numpy()
    D = thetas.shape[1]

    # --- Augmented observations ------------------------------------------------------
    print("Generating augmented observations …")
    xs_list = []
    for i in trange(NUM_DIAG_SIMS, desc="Augment", leave=False):
        xs_list.append(generate_augmented_observation(thetas[i], device=device))
    xs_tensor = torch.stack(xs_list, dim=0)  # [N, MAX_NUM_POINTS, 2]

    # --- Recoverability by SNR and point count -------------------------------------------
    print("Computing recoverability using check_recoverable function...")
    recoverable_mask = np.zeros(NUM_DIAG_SIMS, dtype=bool)
    
    for i in trange(NUM_DIAG_SIMS, desc="Recoverable", leave=False):
        x_event = xs_tensor[i]
        theta = thetas[i]
        
        # Extract times and flux from the observation
        norm_t = x_event[:, 0]  # normalized times [-1,1], padding is -2
        valid = norm_t > -1.5
        times = (norm_t[valid] + 1.0) * (TOTAL_DURATION / 2.0)
        flux = x_event[valid, 1]
        
        # Convert to numpy array for times as expected by check_recoverable
        times_np = times.detach().cpu().numpy()
        
        from utils import RecoverCfg
        cfg = RecoverCfg()
        
        is_recoverable, reason = check_recoverable(theta, times_np, flux, phot_err=0.1, cfg=cfg)
        recoverable_mask[i] = is_recoverable
    
    kept_idx = np.where(recoverable_mask)[0]
    kept = len(kept_idx)
    
    print(f"Combined recoverable: {kept}/{NUM_DIAG_SIMS} ")

    if kept == 0:
        print(f"No recoverable cases under the chosen criteria.")
        return

    # Set event shape from any kept example
    x0 = xs_tensor[kept_idx[0]]
    posterior.set_default_x(x0)
    event_shape = get_event_shape(posterior) or tuple(x0.shape[-2:])
    print("posterior x_event_shape:", event_shape)

    # --- Posterior sampling and outlier scoring -------------------------------------
    print("Running posterior sampling and computing outlier scores …")
    true_thetas_np = theta_np[kept_idx]
    recovered_maps = []
    recovered_medians = []
    all_samples = []
    outlier_scores = []

    with torch.no_grad():
        for j, i in enumerate(trange(kept, desc="Sample", leave=False)):
            idx = kept_idx[j]
            x_i = coerce_x_to_event_shape(xs_tensor[idx], event_shape)
            samples_t = posterior.sample((NUM_POST_SAMPLES,), x=x_i,
                                         show_progress_bars=SHOW_PROGRESS_BARS)
            s_np = samples_t.detach().cpu().numpy()
            
            # Compute MAP estimate (following run_on_datachallenge pattern)
            log_probs = posterior.log_prob(samples_t, x=x_i)
            best_fit_idx = torch.argmax(log_probs)
            map_params = samples_t[best_fit_idx].cpu().numpy()
            
            # Store results
            recovered_median = np.median(s_np, axis=0)
            recovered_maps.append(map_params)
            recovered_medians.append(recovered_median)
            all_samples.append(s_np)
            
            # Compute outlier score using MAP instead of median
            score = compute_outlier_score(true_thetas_np[j], map_params, s_np, OUTLIER_METRIC)
            outlier_scores.append(score)

    recovered_maps = np.array(recovered_maps)
    recovered_medians = np.array(recovered_medians)
    outlier_scores = np.array(outlier_scores)

    # --- Find worst cases -----------------------------------------------------------
    worst_indices = np.argsort(outlier_scores)[-NUM_WORST_CASES:][::-1]  # Highest scores first
    print(f"\nWorst {NUM_WORST_CASES} cases (by {OUTLIER_METRIC} score):")
    for i, worst_idx in enumerate(worst_indices):
        original_idx = kept_idx[worst_idx]
        score = outlier_scores[worst_idx]
        print(f"  {i+1}. Case {original_idx}, Score: {score:.4f}")

    # --- Plot worst cases -----------------------------------------------------------
    param_names = PARAM_NAMES if 'PARAM_NAMES' in globals() else [f"θ{d}" for d in range(D)]
    
    # Create subplots: light curves + parameter comparison
    fig = plt.figure(figsize=(20, 4 * NUM_WORST_CASES))
    
    for i, worst_idx in enumerate(worst_indices):
        original_idx = kept_idx[worst_idx]
        true_theta = thetas[original_idx]
        recovered_theta = recovered_maps[worst_idx]  # Use MAP instead of median
        x_event = xs_tensor[original_idx]
        samples = all_samples[worst_idx]
        score = outlier_scores[worst_idx]
        
        # Create subplot layout: light curve, residuals, parameters
        row = i
        ax_lc = plt.subplot2grid((NUM_WORST_CASES, 4), (row, 0), colspan=2)
        ax_res = plt.subplot2grid((NUM_WORST_CASES, 4), (row, 2), colspan=1)
        ax_params = plt.subplot2grid((NUM_WORST_CASES, 4), (row, 3), colspan=1)
        
        # Plot light curve and residuals
        plot_light_curve_and_residuals(x_event, true_theta, recovered_theta, 
                                     ax_lc, ax_res, original_idx)
        
        # Plot parameter comparison
        x_pos = np.arange(D)
        true_vals = true_theta.cpu().numpy()
        rec_vals = recovered_theta
        
        # Error bars from posterior samples
        sample_std = np.std(samples, axis=0)
        
        ax_params.errorbar(x_pos - 0.1, true_vals, fmt='ro', markersize=8, 
                          label='True', capsize=3)
        ax_params.errorbar(x_pos + 0.1, rec_vals, yerr=sample_std, fmt='b^', 
                          markersize=8, label='Recovered (MAP)', capsize=3)
        
        ax_params.set_xticks(x_pos)
        ax_params.set_xticklabels([name.replace('log10_', '') for name in param_names], 
                                 rotation=45, ha='right')
        ax_params.set_ylabel('Parameter value')
        ax_params.set_title(f'Score: {score:.3f}')
        ax_params.legend(fontsize=8)
        ax_params.grid(True, alpha=0.3)
        
        # Add case info
        ax_lc.text(0.02, 0.98, f'Case {original_idx} (rank #{i+1})', 
                  transform=ax_lc.transAxes, va='top', ha='left', 
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_PATH, f"worst_cases_{OUTLIER_METRIC}_snr{MIN_PEAK_SNR:.1f}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    
    # --- Summary statistics of worst cases ------------------------------------------
    print(f"\nSummary of worst {NUM_WORST_CASES}):")
    print(f"Score range: {outlier_scores[worst_indices[-1]]:.4f} to {outlier_scores[worst_indices[0]]:.4f}")

    print("tE of worst cases:" + ", ".join(f"{10**thetas[kept_idx[idx],2].item():.2f}" for idx in worst_indices))
        
    # Parameter-wise analysis using MAP estimates
    for d in range(D):
        true_worst = true_thetas_np[worst_indices, d]
        rec_worst = recovered_maps[worst_indices, d]  # Use MAP instead of median
        bias = np.mean(rec_worst - true_worst)
        rmse = np.sqrt(np.mean((rec_worst - true_worst) ** 2))
        print(f"  {param_names[d]}: bias={bias:.4f}, RMSE={rmse:.4f}")


    print("Done analyzing outliers.")

if __name__ == "__main__":
    main()

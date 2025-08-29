# utils.py
import torch
import numpy as np
import VBMicrolensing
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from model import TransformerEmbeddingNet
from config import *
import os, random
from dataclasses import dataclass

# Instantiate simulator once to be reused globally in this module
VBM = VBMicrolensing.VBMicrolensing()


def simulate_microlensing_event_pytorch(
    params, times, phot_err=0.0, ld=False, ld_coeff=0.53
):
    """
    Core function to simulate a finite-source point-lens microlensing lightcurve.
    Accepts parameters as a list or tensor. NOW INCLUDES BLENDING.
    """
    if isinstance(params, (list, torch.Tensor)):
        t_0, u_0, log10_t_E, log10_rho, blend_fs = (
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
        )
    else:
        raise TypeError("params must be a list or a tensor")

    t_E = 10 ** (log10_t_E)
    rho = 10 ** (log10_rho)

    target_device = params.device if isinstance(params, torch.Tensor) else "cpu"
    if type(times) is np.ndarray:
        times_tensor = torch.tensor(times, dtype=torch.float32, device=target_device)
    elif isinstance(times, torch.Tensor):
        times_tensor = times.to(dtype=torch.float32, device=target_device)
    else:
        raise TypeError("times must be a numpy array or a tensor")

    # Calculate the impact parameter 'u'
    x_arr = (times_tensor - t_0) / t_E
    y_arr = u_0 * torch.ones_like(x_arr)
    u_arr = torch.sqrt(x_arr**2 + y_arr**2)

    # Calculate raw magnification
    with torch.no_grad():
        VBM.a1 = 0.0  # assume no limb darkening for now
        magnification_list = [VBM.ESPLMag2(u.item(), rho.item()) for u in u_arr]
        magnification = torch.tensor(magnification_list, device=target_device)

    # Apply blending
    normalized_flux = blend_fs * magnification + (1.0 - blend_fs)

    # Add Gaussian noise to the final blended flux
    noisy_flux = normalized_flux + torch.randn_like(normalized_flux) * phot_err

    return noisy_flux


def generate_observation_times(
    num_gaps_range=AUG_NUM_GAPS_RANGE,
    gap_length_range=AUG_GAP_LENGTH_RANGE,
    dropout_rate_range=AUG_DROPOUT_RATE_RANGE,
):
    """
    Generates a realistic, variable time array for data augmentation.

    This function simulates a survey by:
    1. Starting with a very dense set of potential observations.
    2. Introducing a variable number of seasonal-like gaps of variable length.
    3. Applying a random dropout to the remaining points to simulate weather, etc.
    4. Sub-sampling to the maximum number of points allowed by the network.

    Args:
        num_gaps_range (tuple): The (min, max) number of large gaps to introduce.
        gap_length_range (tuple): The (min, max) length in days for each gap.
        dropout_rate (float): The fraction of remaining points to randomly drop.

    Returns:
        np.ndarray: An array of observation times.
    """
    # Start with a very dense, uniform set of potential observations (e.g., 5 per day)
    # This provides a rich base from which to remove points.
    # base_times = np.linspace(0, TOTAL_DURATION, int(TOTAL_DURATION * 5))
    base_times = np.linspace(0, TOTAL_DURATION, NUM_POINTS_BASE)

    # --- 1. Introduce Large Gaps ---
    keep_mask = np.ones_like(base_times, dtype=bool)
    num_gaps = np.random.randint(num_gaps_range[0], num_gaps_range[1] + 1)

    for _ in range(num_gaps):
        gap_length = np.random.uniform(gap_length_range[0], gap_length_range[1])
        # Ensure the gap starts at a valid time
        gap_start = np.random.uniform(0, TOTAL_DURATION - gap_length)

        # Mark points within the gap for removal
        gap_indices = (base_times >= gap_start) & (base_times <= gap_start + gap_length)
        keep_mask[gap_indices] = False

    times_after_gaps = base_times[keep_mask]

    dropout_rate = np.random.uniform(dropout_rate_range[0], dropout_rate_range[1])

    # --- 2. Apply Random Dropout ---
    if dropout_rate > 0 and len(times_after_gaps) > 0:
        num_to_keep = int(len(times_after_gaps) * (1 - dropout_rate))
        chosen_indices = np.random.choice(
            len(times_after_gaps), num_to_keep, replace=False
        )
        times_after_dropout = np.sort(times_after_gaps[chosen_indices])
    else:
        times_after_dropout = times_after_gaps

    final_times = times_after_dropout

    # --- 3. Final Sub-sampling to MAX_NUM_POINTS ---
    # Ensure the number of points doesn't exceed the max padding length
    if len(final_times) > MAX_NUM_POINTS:
        idx = np.random.choice(len(final_times), MAX_NUM_POINTS, replace=False)
        final_times = np.sort(final_times[idx])

    # Handle the edge case of generating an empty lightcurve
    if len(final_times) == 0:
        # If all points were removed, generate a few random points to avoid errors
        return np.sort(np.random.rand(10) * TOTAL_DURATION)

    return final_times


def generate_base_simulation(theta):
    """
    Generates a single, full-cadence, NOISELESS lightcurve for the master dataset.
    """
    times = np.linspace(0, TOTAL_DURATION, NUM_POINTS_BASE)

    # flux will now be on the same device as theta
    flux = simulate_microlensing_event_pytorch(theta, times, phot_err=BASE_SIM_NOISE)

    # Ensure normalized_times is created on the same device as theta/flux
    normalized_times = torch.tensor(
        times / (TOTAL_DURATION / 2.0) - 1.0, dtype=torch.float32, device=theta.device
    )

    return torch.stack((normalized_times, flux), dim=1)


class MicrolensingAugmentationDataset(torch.utils.data.Dataset):
    """
    Generates a fresh augmented light-curve every time __getitem__ is called.
    All work happens on *CPU*; the training loop later moves complete batches
    to GPU, so DataLoader workers never touch CUDA.
    """

    def __init__(
        self,
        thetas: torch.Tensor,
        n_augs: int = 1,
        *,
        num_gaps_range=AUG_NUM_GAPS_RANGE,
        gap_length_range=AUG_GAP_LENGTH_RANGE,
        dropout_rate_range=AUG_DROPOUT_RATE_RANGE,
    ):
        super().__init__()
        # -------- keep base θ on CPU so worker subprocesses are CUDA-free ---- #
        self.thetas = thetas.cpu()
        self.n_augs = n_augs

        self.num_gaps_range = num_gaps_range
        self.gap_length_range = gap_length_range
        self.dropout_rate_range = dropout_rate_range

    # ----------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.thetas) * self.n_augs  # virtual size per epoch

    def __getitem__(self, idx):
        base_idx = idx // self.n_augs
        theta = self.thetas[base_idx]  # CPU tensor

        MAX_TRIES = 5
        last_padded = None

        for _ in range(MAX_TRIES):
            # 1) sample times and noise
            times = generate_observation_times(
                num_gaps_range=self.num_gaps_range,
                gap_length_range=self.gap_length_range,
                dropout_rate_range=self.dropout_rate_range,
            )
            phot_err = np.random.uniform(PHOT_ERR_MIN, PHOT_ERR_MAX)

            # 2) simulate flux (CPU path)
            flux = simulate_microlensing_event_pytorch(theta, times, phot_err=phot_err)

            # 3) recoverability gate
            ok, _reason = check_recoverable(theta, times, flux, phot_err)
            # Optional: collect stats here if you want

            # 4) pack + pad
            norm_times = torch.tensor(
                times / (TOTAL_DURATION / 2.0) - 1.0, dtype=torch.float32
            )
            sigma_vec = torch.full((len(times),), float(phot_err), dtype=torch.float32)
            data = torch.stack(
                (norm_times, flux.to(dtype=torch.float32), sigma_vec), dim=1
            )  # (n, 3)

            padded = torch.full(
                (MAX_NUM_POINTS, 3), -2.0
            )  # init with -2; then set channels
            padded[:, 1] = 0.0
            padded[:, 2] = 0.0  # sigma channel

            n = min(len(times), MAX_NUM_POINTS)
            padded[:n] = data[:n]
            last_padded = padded.float(), theta.float()

            if ok:
                return last_padded

        # fallback if none accepted after MAX_TRIES
        return last_padded


def generate_augmented_observation(
    theta,
    device="cpu",
    num_gaps_range=AUG_NUM_GAPS_RANGE,
    gap_length_range=AUG_GAP_LENGTH_RANGE,
    dropout_rate_range=AUG_DROPOUT_RATE_RANGE,
):
    """
    Generates a single, randomly augmented observation for testing.
    This mimics the data the network was trained on.
    """

    # Generate times
    times = generate_observation_times(
        num_gaps_range=num_gaps_range,
        gap_length_range=gap_length_range,
        dropout_rate_range=dropout_rate_range,
    )
    phot_err = np.random.uniform(PHOT_ERR_MIN, PHOT_ERR_MAX)
    flux = simulate_microlensing_event_pytorch(theta, times, phot_err=phot_err).to(
        device
    )

    num_obs = len(times)
    normalized_times = torch.tensor(
        times / (TOTAL_DURATION / 2.0) - 1.0, dtype=torch.float32, device=device
    )
    sigma_vec = torch.full(
        (len(times),), float(phot_err), dtype=torch.float32, device=device
    )
    data = torch.stack(
        (normalized_times, flux.to(dtype=torch.float32), sigma_vec), dim=1
    )

    padded_data = torch.full((MAX_NUM_POINTS, 3), fill_value=-2.0, device=device)
    padded_data[:, 1] = 0.0
    padded_data[:, 2] = 0.0
    padded_data[:num_obs, :] = data

    return padded_data


def prepare_real_data_for_network(
    times, flux, device, t_start_window=None, errors=None
):
    """
    Pads and formats a real lightcurve to be fed into the transformer network.
    This version expects numpy arrays for time and flux.
    """
    # 1. Select a window of TOTAL_DURATION around the peak

    if t_start_window is None:
        peak_time_estimate = times[np.argmax(flux)]
        t_start_window = peak_time_estimate - (TOTAL_DURATION / 2.0)

    mask = (times >= t_start_window) & (times <= t_start_window + TOTAL_DURATION)

    if not np.any(mask):
        print("ERROR: No data points found within the analysis window.")
        return None, None

    times_window = times[mask]
    flux_window = flux[mask]

    if errors is None:  # if not provided, calculate photometric errors here
        base_mask = np.ones_like(flux_window, dtype=bool)
        med = np.median(flux_window[base_mask])
        sig = np.std(flux_window[base_mask])
        errors_vec = np.full_like(flux_window, max(1e-6, sig))
    else:
        errors_vec = errors[mask]  # expect same shape as flux in 'times' space

    # 2. Normalize time into the [-1, 1] range for the network
    times_sim_coords = times_window - t_start_window
    normalized_times_net = (times_sim_coords / (TOTAL_DURATION / 2.0)) - 1.0

    # 3. Pad the data
    num_obs = len(times_window)
    if num_obs > MAX_NUM_POINTS:
        print(f"Warning: Found {num_obs} points, truncating to {MAX_NUM_POINTS}.")
        num_obs = MAX_NUM_POINTS
        idx = np.sort(
            np.random.choice(len(times_window), MAX_NUM_POINTS, replace=False)
        )
        times_sim_coords = times_sim_coords[idx]
        normalized_times_net = normalized_times_net[idx]
        errors_vec = errors_vec[idx]
        flux_window = flux_window[idx]

    time_tensor = torch.tensor(normalized_times_net, dtype=torch.float32, device=device)
    flux_tensor = torch.tensor(flux_window, dtype=torch.float32, device=device)
    sigma_tensor = torch.tensor(errors_vec, dtype=torch.float32, device=device)

    net_input_data = torch.full((MAX_NUM_POINTS, 3), fill_value=-2.0, device=device)
    net_input_data[:, 1] = 0.0
    net_input_data[:, 2] = 0.0
    net_input_data[:num_obs, 0] = time_tensor
    net_input_data[:num_obs, 1] = flux_tensor
    net_input_data[:num_obs, 2] = sigma_tensor

    return net_input_data, t_start_window


def compute_peak_snr(x_event: torch.Tensor, theta: torch.Tensor) -> float:
    """Computes peak signal-to-noise ratio of a microlensing event."""
    flux = x_event[:, 1].cpu().numpy()
    norm_times = x_event[:, 0].cpu().numpy()
    valid_mask = norm_times > -1.5
    flux_valid = flux[valid_mask]

    if len(flux_valid) < 5:
        return 0.0

    times_original = (norm_times[valid_mask] + 1.0) * (TOTAL_DURATION / 2.0)
    t0 = float(theta[0].cpu())
    tE = float(10.0 ** theta[2].cpu())

    # Define baseline region (e.g., > 2*tE away from peak)
    baseline_mask = np.abs(times_original - t0) > 2.0 * tE

    if np.sum(baseline_mask) < 3:  # Not enough baseline points
        baseline_flux = flux_valid[flux_valid < np.percentile(flux_valid, 60)]
    else:
        baseline_flux = flux_valid[baseline_mask]

    if len(baseline_flux) < 2:
        return 0.0

    baseline_median = np.median(baseline_flux)
    noise_std = np.std(baseline_flux)
    if noise_std < 1e-9:
        return 0.0

    peak_flux = np.max(flux_valid)
    snr = (peak_flux - baseline_median) / noise_std

    return max(0.0, snr)


def compute_peak_snr_exact(x_event: torch.Tensor, theta: torch.Tensor) -> float:
    """Computes peak signal-to-noise ratio of a microlensing event."""
    flux = x_event[:, 1].cpu().numpy()
    norm_times = x_event[:, 0].cpu().numpy()
    valid_mask = norm_times > -1.5
    flux_valid = flux[valid_mask]

    if len(flux_valid) < 5:
        return 0.0

    times_original = (norm_times[valid_mask] + 1.0) * (TOTAL_DURATION / 2.0)
    t0 = float(theta[0].cpu())
    tE = float(10.0 ** theta[2].cpu())
    u0 = float(theta[1].cpu())
    rho = float(10.0 ** theta[3].cpu())
    blend_fs = float(theta[4].cpu())

    # Define baseline region (e.g., > 2*tE away from peak)
    baseline_mask = np.abs(times_original - t0) > 2.0 * tE

    if np.sum(baseline_mask) < 3:  # Not enough baseline points
        baseline_flux = flux_valid[flux_valid < np.percentile(flux_valid, 60)]
    else:
        baseline_flux = flux_valid[baseline_mask]

    if len(baseline_flux) < 2:
        return 0.0

    baseline_median = np.median(baseline_flux)
    noise_std = np.std(baseline_flux)
    if noise_std < 1e-9:
        return 0.0

    peak_magnification = VBM.ESPLMag2(u0, rho)
    blended_peak_magnification = blend_fs * peak_magnification + (1.0 - blend_fs)
    snr = (blended_peak_magnification - baseline_median) / noise_std

    return max(0.0, snr)


def ensure_dir(path_or_file: str):
    """
    Create the directory for a file path or the path itself if it's a dir.
    Safe to call repeatedly.
    """
    path = path_or_file
    # if it's a file path, create its parent
    if os.path.splitext(path)[1]:
        path = os.path.dirname(path)
    if path:
        os.makedirs(path, exist_ok=True)


def set_global_seeds(seed: int = 0, deterministic: bool = True):
    """Make runs reproducible."""
    import numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_posterior_bundle(
    flow, prior, embed_cfg: dict, path: str, meta: dict | None = None
):
    """
    Save just what's needed to rebuild the posterior later.
    Compatible with build_posterior_from_saved(...).
    """
    import subprocess, sys

    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_rev = "unknown"

    meta_full = {
        "version": 1,
        "git": git_rev,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    if meta:
        meta_full.update(meta)

    bundle = {
        "prior": prior,  # serialized safely
        "flow_state_dict": flow.state_dict(),
        "embed_cfg": embed_cfg,  # transformer hyperparams
        "embed_state_dict": (
            flow.embedding_net.state_dict() if hasattr(flow, "embedding_net") else None
        ),
        "builder_kwargs": {
            "model": "maf",
            "z_score_x": "none",
            "z_score_theta": "independent",
        },
        "meta": meta_full,
    }
    torch.save(bundle, path)


def load_saved_cpu(path: str):
    return torch.load(path, map_location="cpu")


def _prior_to_device(prior, device: str):
    # Handle BoxUniform across SBI versions
    if isinstance(prior, BoxUniform):
        low = getattr(prior, "low", None)
        high = getattr(prior, "high", None)

        # Older/newer SBI may store bounds on base_dist
        if (low is None or high is None) and hasattr(prior, "base_dist"):
            bd = prior.base_dist
            low = getattr(bd, "low", low)
            high = getattr(bd, "high", high)

        if low is None or high is None:
            raise AttributeError(
                "Could not find low/high on BoxUniform (neither on object nor base_dist)."
            )
        return BoxUniform(low=low.to(device), high=high.to(device))

    # Generic fallback for other torch distributions
    try:
        return prior.to(device)
    except Exception:
        if hasattr(prior, "loc") and hasattr(prior, "scale"):
            loc = prior.loc.to(device)
            scale = prior.scale.to(device)
            return type(prior)(loc, scale)
        raise RuntimeError(
            f"Don’t know how to move prior of type {type(prior)} to {device}"
        )


def build_posterior_from_saved(saved: dict, device: str = "cpu"):
    prior_cpu = saved["prior"]
    prior = _prior_to_device(prior_cpu, device)

    cfg = saved["embed_cfg"]
    input_dim = cfg.get("INPUT_DIM", 3)  # Default to 3D input (time, flux, sigma)
    embed = TransformerEmbeddingNet(
        d_model=cfg["D_MODEL"],
        nhead=cfg["N_HEAD"],
        d_hid=cfg["D_HID"],
        nlayers=cfg["N_LAYERS"],
        dropout=cfg["DROPOUT"],
        input_dim=input_dim,
    )
    if saved.get("embed_state_dict") is not None:
        embed.load_state_dict(saved["embed_state_dict"])

    builder = posterior_nn(embedding_net=embed, **saved["builder_kwargs"])

    # Make the dummy tensors on the target device too
    dummy_theta = prior.sample((64,)).to(device)
    dummy_x = torch.zeros(64, MAX_NUM_POINTS, input_dim, device=device)
    flow = builder(dummy_theta, dummy_x)
    flow.load_state_dict(saved["flow_state_dict"])
    flow.to(device)

    npe = NPE(prior=prior, density_estimator=builder, device=device)
    posterior = npe.build_posterior(flow)
    return prior, posterior


@dataclass
class RecoverCfg:
    n_peak_min: int = 5  # at least this many samples within |t - t0| <= peak_k * tE
    n_base_min: int = 5  # at least this many baseline samples
    peak_k: float = 0.5  # "peak" half-width in units of tE
    base_k: float = 2.0  # baseline region is |t - t0| > base_k * tE
    snr_min: float = 3.0  # require peak SNR >= this
    min_points: int = 10  # absolute minimum #points in the window


def _counts_peak_and_baseline(times: np.ndarray, theta: torch.Tensor, cfg: RecoverCfg):
    t0 = float(theta[0])
    tE = float(10.0 ** theta[2])
    if tE <= 0.0:
        return 0, 0
    in_peak = np.abs(times - t0) <= cfg.peak_k * tE
    in_base = np.abs(times - t0) >= cfg.base_k * tE
    return int(in_peak.sum()), int(in_base.sum())


def check_recoverable(theta, times, flux, phot_err, cfg: RecoverCfg = RecoverCfg()):

    if len(times) < cfg.min_points:
        return False, "too_few_points"

    n_peak, n_base = _counts_peak_and_baseline(times, theta, cfg)
    if n_peak < cfg.n_peak_min:
        return False, "no_peak_samples"
    if n_base < cfg.n_base_min:
        return False, "no_baseline"

    # robust SNR using only true baseline region
    base_mask = np.abs(times - float(theta[0])) >= cfg.base_k * (
        10.0 ** float(theta[2])
    )
    if base_mask.sum() >= 5:
        base_flux = flux.detach().cpu().numpy()[base_mask]
        noise_std = base_flux.std()
        if noise_std <= 0:
            return False, "zero_noise_std"
        peak_mag = VBMicrolensing.VBMicrolensing().ESPLMag2(
            float(theta[1]), float(10.0 ** theta[3])
        )
        snr = (peak_mag - base_flux.mean()) / noise_std
        if snr < cfg.snr_min:
            return False, "low_snr"

    return True, "ok"

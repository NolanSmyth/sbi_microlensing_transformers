# 4_analyze_kmt_2073.py
import os, re, glob
import dill
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from tqdm import tqdm

# import VBMicrolensing
# VBM = VBMicrolensing.VBMicrolensing()


# project imports
from config import (
    FIGURES_PATH,
    LOADED_POSTERIOR_PATH,
    PARAM_NAMES,
    TOTAL_DURATION,
    MAX_NUM_POINTS,
    KMT_DATA_DIR,
)
from utils import (
    prepare_real_data_for_network,
    simulate_microlensing_event_pytorch,
    load_saved_cpu,
    build_posterior_from_saved,
)

# --------------------------
# Plot settings / windows
# --------------------------
data_dir = KMT_DATA_DIR
t0p = 8708.598  # HJD' peak for KMT-2019-BLG-2073
x_top = (8705.0, 8712.0)  # ~5-day window
x_zoom = (8707.5, 8709.5)  # zoom


# --------------------------
# Helpers
# --------------------------
def read_I_pysis(path: str) -> pd.DataFrame:
    """pySIS I-band format: HJD'  Δflux  flux_err  mag  mag_err  fwhm  sky  secz"""
    cols = ["HJDp", "dflux", "eflux", "mag", "emag", "fwhm", "sky", "secz"]
    df = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None, names=cols, engine="python"
    )
    df["file"] = os.path.basename(path)
    return df


def site_field_label(filename: str) -> str:
    """Turn KMTA/KMTC/KMTS + field into labels like CTIO(02), SAAO(42), SSO(02)."""
    u = os.path.basename(filename).upper()
    m = re.search(r"KMT([ACS])(\d{2})_I\.PYSIS$", u)
    if m:
        site_code = m.group(1)
        field = m.group(2)
        site = {"C": "CTIO", "S": "SAAO", "A": "SSO"}[site_code]
        return f"{site}({field})"
    site = (
        "CTIO"
        if "KMTC" in u
        else ("SAAO" if "KMTS" in u else ("SSO" if "KMTA" in u else "KMT"))
    )
    field = "02" if "02" in u else ("42" if "42" in u else "")
    return f"{site}({field})" if field else site


def normalize_per_file(group, t0p):
    # Per-dataset baseline from off-peak points
    far = group[np.abs(group["HJDp"] - t0p) > 2.0]
    base_mag = (
        np.nanmedian(far["mag"].values)
        if len(far) >= 20
        else np.nanmedian(group["mag"].values)
    )
    print("group" + str(group["file"].values[0]) + " baseline mag: " + str(base_mag))
    g = group.copy()
    g["flux_rel"] = 10 ** (-0.4 * (g["mag"] - base_mag))  # baseline ~ 1
    g["eflux_rel"] = (
        g["flux_rel"] * (np.log(10) / 2.5) * g["emag"]
    )  # propagate mag errors
    g["base_mag"] = base_mag
    return g


# --- corner overlays (truth ±σ) ---
def _axes_grid_from_fig(fig, K):
    grid = [[None] * K for _ in range(K)]
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        r = ss.rowspan.start
        c = ss.colspan.start
        if 0 <= r < K and 0 <= c < K:
            grid[r][c] = ax
    return grid


def add_truth_gaussians_on_corner(fig, truths, sigmas, labels):
    """
    Overlay Gaussian 'truth' bands on a corner plot, robust to axes order.
    - Diagonals: vertical line at mean and ±1σ shaded band.
    - Off-diagonals (lower triangle only): 1σ rectangle + crosshairs.
    """
    truths = np.asarray(truths, float)
    sigmas = np.asarray(sigmas, float)
    K = len(labels)

    A = _axes_grid_from_fig(fig, K)

    # Diagonals
    for i in range(K):
        ax = A[i][i]
        if ax is None:
            continue
        mu, sd = truths[i], sigmas[i]
        ax.axvline(mu, color="k", lw=1.2, alpha=0.9)
        if np.isfinite(sd) and sd > 0:
            ax.axvspan(mu - sd, mu + sd, alpha=0.20, ec=None)


# --- convert linear sigma -> log10 sigma ---
def sigma_linear_to_log10(mu_linear, sigma_linear):
    # σ_log10 ≈ σ_linear / (mu_linear * ln 10)
    return float(sigma_linear) / (float(mu_linear) * np.log(10))


# --------------------------
# Load all I-band pySIS
# --------------------------
paths = sorted(glob.glob(os.path.join(data_dir, "*_I.pysis")))
if not paths:
    raise FileNotFoundError(f"No *_I.pysis files found in {data_dir}")

frames = []
for p in paths:
    df = read_I_pysis(p)
    df["label"] = site_field_label(p)
    frames.append(df)
I = pd.concat(frames, ignore_index=True)

# =========================================================
# Posterior estimation (corner + predictive)
# =========================================================
print("=== Posterior estimation on KMT-2019-BLG-2073 (I band) ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Global baseline (flux for modeling)
off_peak_mask = np.abs(I["HJDp"] - t0p) > 2.0
baseline_data = I[off_peak_mask]["mag"]


baseline_mag_global = 18.15  # the “I=18” convention


# absolute fluxes in the I=18 system
I["flux_abs"] = 10 ** (-0.4 * (I["mag"] - baseline_mag_global))
I["flux_err_abs"] = I["flux_abs"] * (np.log(10) / 2.5) * I["emag"]

baseline_flux = I["flux_abs"][off_peak_mask].median()
baseline_mag_data = -2.5 * np.log10(baseline_flux) + baseline_mag_global
print("Global baseline flux (I=18 system):", baseline_flux)

flux = I["flux_abs"].values / baseline_flux
flux_err = I["flux_err_abs"].values
times = I["HJDp"].values

print(np.median(flux))
# per-file baseline magnitude from off-peak points (e.g., |t - t0| > 2 d)
# base_mag_by_file = (
#     I.loc[np.abs(I['HJDp'] - t0p) > 2.0]
#      .groupby('file')['mag']
#      .median()
#      .rename('base_mag')
# )
# I = I.merge(base_mag_by_file, on='file', how='left')

# # per-file baseline flux in the I=18 system
# I['f_base_abs'] = 10**(-0.4 * (I['base_mag'] - baseline_mag_global))

# # relative flux (baseline ≈ 1) and its error
# I['flux_rel']     = I['flux_abs'] / I['f_base_abs']
# I['eflux_rel']    = I['flux_err_abs'] / I['f_base_abs']

# # Use these for inference/plots:
# flux  = I['flux_rel'].values
# flux_err = I['eflux_rel'].values
# times = I['HJDp'].values

# for name, g in I.groupby('file'):
#     off = np.abs(g['HJDp'] - t0p) > 2
#     print(name, "baseline ~", np.median(g.loc[off, 'flux_rel']))

print(f"Before outlier removal: {len(flux)} points")
flux_err_median = np.median(flux_err)
flux_err_mad = np.median(np.abs(flux_err - flux_err_median))
flux_err_threshold = flux_err_median + 5 * flux_err_mad
good_err_mask = flux_err < flux_err_threshold

flux_median = np.median(flux)
flux_mad = np.median(np.abs(flux - flux_median))
flux_lower = flux_median - 20 * flux_mad
flux_upper = flux_median + 50 * flux_mad
good_flux_mask = (flux >= flux_lower) & (flux <= flux_upper)

# Select the 1000 data points closest to the peak time t0p
if len(times) > MAX_NUM_POINTS:
    time_diffs = np.abs(times - t0p)
    closest_indices = np.argsort(time_diffs)[:MAX_NUM_POINTS]
    window_mask = np.zeros_like(times, dtype=bool)
    window_mask[closest_indices] = True
else:
    window_mask = np.ones_like(times, dtype=bool)

outlier_mask = good_err_mask & good_flux_mask & window_mask
# outlier_mask = window_mask & good_err_mask

flux = flux[outlier_mask]
flux_err = flux_err[outlier_mask]
times = times[outlier_mask]

print(f"After outlier removal: {len(flux)} points (removed {np.sum(~outlier_mask)})")
print(f"Flux range after cleaning: {flux.min():.3f} to {flux.max():.3f}")

# Prepare data for the network
print("Preparing data for the network...")

print("before", (t0p - TOTAL_DURATION / 2.0))

net_input, t_start_window = prepare_real_data_for_network(
    times, flux, device, t_start_window=(t0p - TOTAL_DURATION / 2.0), errors=flux_err
)

print("after", (t_start_window))


print(f"{t_start_window} is the start of the time window.")
print(f"{t0p} is the peak time of the event.")
if net_input is None:
    raise RuntimeError(
        "prepare_real_data_for_network returned None; please check inputs."
    )

# Load posterior
print(f"Loading trained posterior from {LOADED_POSTERIOR_PATH} ...")
saved = load_saved_cpu(LOADED_POSTERIOR_PATH)
_, posterior = build_posterior_from_saved(saved, device=device)

# Sample posterior
NUM_SAMPLES = 5000
print(f"Sampling {NUM_SAMPLES} draws...")
with torch.no_grad():
    samples = posterior.sample((NUM_SAMPLES,), x=net_input, show_progress_bars=True)

# MAP & median
print("Computing MAP and median...")
logp = posterior.log_prob(samples, x=net_input)
best_idx = torch.argmax(logp)
map_params = samples[best_idx]
median_params = samples.median(dim=0).values
print("MAP:", np.round(map_params.cpu().numpy(), 6))
print("Median:", np.round(median_params.cpu().numpy(), 6))

# --------------------------
# "Truth" parameters + σ
# --------------------------
# Parameter order: [t0 (relative), u0, log10 tE, log10 rho, fs]
# truths = [t0p - t_start_window, 0.241, np.log10(0.267), np.log10(1.184), 1.0]
# truths = [t0p - t_start_window, 0.163, np.log10(0.272), np.log10(1.138), 1.0] #from paper, using tlc pipeline
truths = [
    8708.58092 - t_start_window,
    0.324,
    np.log10(0.5),
    np.log10(0.01),
    0.61,
]  # pysis raw
# fs_fit = 0.947
# fs_fit = 0.873
# truths = [t0p - t_start_window, 0.324, np.log10(0.267), np.log10(0.9), 0.947]

# Given linear one-sigma for tE and rho (from the paper):
sigma_tE_linear = 0.007
sigma_rho_linear = 0.012
sigma_logtE = sigma_linear_to_log10(10 ** truths[2], sigma_tE_linear)
sigma_logrho = sigma_linear_to_log10(10 ** truths[3], sigma_rho_linear)

# Your other sigmas (t0, u0, fs) — edit if you have better values:
sigmas_truth = np.array([0.005, 0.103, sigma_logtE, sigma_logrho, 0.20], dtype=float)

print("Truth (mean):", np.round(truths, 6))
print("Truth sigmas:", np.round(sigmas_truth, 6))

# --------------------------
# Corner plot + Gaussian truth overlays
# --------------------------
print("Saving corner plot with truth Gaussians...")
fig_corner = corner.corner(
    samples.cpu().numpy(),
    labels=PARAM_NAMES,
    color="#0072B2",
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 16},
    label_kwargs={"fontsize": 16},
    # range = [(9.9,10.1), (0,0.6), (-0.8,0.0), (-1.2,0.2), (0,1)],
    levels=[0.393, 0.864],
    plot_datapoints=False,
    smooth=True,
    fill_contours=True,
)
# add_truth_gaussians_on_corner(fig_corner, truths, sigmas_truth, PARAM_NAMES)
fig_corner.suptitle("Posterior — KMT-2019-BLG-2073 (I band)", fontsize=14, y=1.02)
fig_corner.savefig(
    os.path.join(FIGURES_PATH, "kmt_2019_blg_2073_corner.pdf"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.0,
)
plt.close(fig_corner)

# =========================================================
# Predictive bands: posterior + TRUTH-GAUSSIAN band
# =========================================================
print("Building light-curve bands...")

# Dense grid in the network's normalized time
dense = np.linspace(0, TOTAL_DURATION, 1000)
dense_norm = dense
t_axis = t_start_window + dense

# Posterior predictive (already have samples)
Kpost = min(1000, NUM_SAMPLES)
idx = np.random.choice(NUM_SAMPLES, Kpost, replace=False)
preds_post = []
with torch.no_grad():
    for k in idx:
        theta_k = samples[k]
        preds_post.append(
            simulate_microlensing_event_pytorch(theta_k, dense_norm).cpu().numpy()
        )
preds_post = np.stack(preds_post, axis=0)
p16_post, p84_post = np.percentile(preds_post, [16, 84], axis=0)
p5_post, p95_post = np.percentile(preds_post, [5, 95], axis=0)

# Truth-parameter predictive band (Gaussian around reported values)
Ktruth = 1000
rng = np.random.default_rng(0)
truths_mean = np.array(truths, dtype=float)
truths_cov = np.diag(
    sigmas_truth**2
)  # diagonal Gaussian; add covariances here if you have them
theta_truth_samples = rng.multivariate_normal(truths_mean, truths_cov, size=Ktruth)

# (Optional) sanity guards — e.g., keep fs>0
theta_truth_samples[:, 4] = np.clip(theta_truth_samples[:, 4], 0.01, None)

preds_truth = []
with torch.no_grad():
    for th in theta_truth_samples:
        th_torch = torch.tensor(th, dtype=torch.float32, device=device)
        preds_truth.append(
            simulate_microlensing_event_pytorch(th_torch, dense_norm, ld=False)
            .cpu()
            .numpy()
        )  # limb darkening on here.
# preds_truth = (np.stack(preds_truth, axis=0) - (1-fs_fit))/fs_fit  # scale by fs_fit to match the data flux
preds_truth = np.stack(preds_truth, axis=0)  # scale by fs_fit to match the data flux

p16_truth, p84_truth = np.percentile(preds_truth, [16, 84], axis=0)

# Also compute MAP & reported (mean) single curves in flux space
with torch.no_grad():
    lc_map = simulate_microlensing_event_pytorch(map_params, dense_norm).cpu().numpy()
    # lc_reported =((simulate_microlensing_event_pytorch(torch.tensor(truths_mean, dtype=torch.float32, device=device), dense_norm, ld=True).cpu().numpy()) - (1-fs_fit))/fs_fit
    lc_reported = (
        simulate_microlensing_event_pytorch(
            torch.tensor(truths_mean, dtype=torch.float32, device=device),
            dense_norm,
            ld=False,
        )
        .cpu()
        .numpy()
    )


# ---------------------------------------------------------
# Plot in RELATIVE FLUX (like before), with both bands
# ---------------------------------------------------------
fig, axes = plt.subplots(
    2,
    1,
    figsize=(8, 5),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)

# Top panel: main plot
ax_main = axes[0]
ax_main.errorbar(
    times, flux, yerr=flux_err, fmt=".", ms=4, alpha=0.6, label="Data", color="k"
)

# bands
# ax_main.fill_between(t_axis, p16_post,  p84_post,  alpha=0.25, label="Posterior 16-84%", color="C1")
ax_main.fill_between(t_axis, p5_post, p95_post, alpha=0.25, color="#0072B2")
# ax_main.fill_between(t_axis, p16_truth, p84_truth, alpha=0.20, label="Truth-params 16-84%", color="k")

# means
ax_main.plot(t_axis, lc_map, lw=3, label="NPE", color="#0072B2")
ax_main.plot(t_axis, lc_reported, lw=3, ls="--", color="#D55E00", label="Reported PSPL")

ax_main.set_xlim(x_top)
ax_main.set_ylim(0.8, 2.4)
ax_main.set_ylabel("Relative Flux", fontsize=14)
ax_main.grid(ls="--", alpha=0.3)
ax_main.legend(fontsize=8)
plt.tight_layout()


# Bottom panel: residuals
ax_resid = axes[1]

# Calculate model directly at data time points for residual calculation
times_norm = times - t_start_window
with torch.no_grad():
    # MAP model
    lc_map_at_data = (
        simulate_microlensing_event_pytorch(map_params, times_norm).cpu().numpy()
    )

    # Reported (truth) model
    truths_mean_torch = torch.tensor(truths_mean, dtype=torch.float32, device=device)
    lc_reported_at_data_raw = (
        simulate_microlensing_event_pytorch(truths_mean_torch, times_norm, ld=False)
        .cpu()
        .numpy()
    )
    # lc_reported_at_data = (lc_reported_at_data_raw - (1 - fs_fit)) / fs_fit
    lc_reported_at_data = lc_reported_at_data_raw

# Calculate residuals
residuals_map = flux - lc_map_at_data
residuals_reported = flux - lc_reported_at_data

# Plot residuals
ax_resid.errorbar(
    times,
    residuals_map,
    yerr=flux_err,
    fmt=".",
    ms=4,
    alpha=0.5,
    color="#0072B2",
    label="NPE MAP",
    elinewidth=0.5,
    zorder=2,
)
ax_resid.errorbar(
    times,
    residuals_reported,
    yerr=flux_err,
    fmt=".",
    ms=4,
    alpha=0.6,
    color="#D55E00",
    label="Reported PSPL",
    marker="^",
    elinewidth=0.5,
    zorder=1,
)

# Zero line
ax_resid.axhline(0, color="gray", linestyle="-", alpha=0.5)

ax_resid.set_xlim(x_top)
ax_resid.set_xlabel("HJD - 2450000", fontsize=14)
ax_resid.set_ylabel("Residuals", fontsize=14)
ax_resid.grid(ls="--", alpha=0.3)
ax_resid.legend(fontsize=8)

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_PATH, "kmt_2019_blg_2073_fit_predictive_flux.pdf"),
    bbox_inches="tight",
    pad_inches=0.0,
)
plt.close(fig)

# ---------------------------------------------------------
# Plot in MAGNITUDE space: data + truth mean + truth band
# ---------------------------------------------------------
print("Plotting truth model in magnitude space with band...")
# Convert truth band and mean to magnitudes using global baseline
truth_mag_p16 = baseline_mag_data - 2.5 * np.log10(np.clip(p16_truth, 1e-6, None))
truth_mag_p84 = baseline_mag_data - 2.5 * np.log10(np.clip(p84_truth, 1e-6, None))
truth_mag_mean = baseline_mag_data - 2.5 * np.log10(np.clip(lc_reported, 1e-6, None))
model_mag_map = baseline_mag_data - 2.5 * np.log10(np.clip(lc_map, 1e-6, None))
model_mag_p5 = baseline_mag_data - 2.5 * np.log10(np.clip(p5_post, 1e-6, None))
model_mag_p95 = baseline_mag_data - 2.5 * np.log10(np.clip(p95_post, 1e-6, None))

fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=False)

# Top panel (full window)
ax = axes[0]
cut = I[(I["HJDp"] >= x_top[0]) & (I["HJDp"] <= x_top[1])]
ax.errorbar(
    I["HJDp"],
    I["mag"],
    yerr=I["emag"],
    fmt=".",
    ms=3,
    elinewidth=0.6,
    capsize=0,
    alpha=0.7,
)

mask_top = (t_axis >= x_top[0]) & (t_axis <= x_top[1])
# ax.fill_between(t_axis[mask_top], truth_mag_p16[mask_top], truth_mag_p84[mask_top], alpha=0.2, label="Truth 16-84% (mag)")
ax.fill_between(
    t_axis[mask_top],
    model_mag_p5[mask_top],
    model_mag_p95[mask_top],
    alpha=0.2,
    color="C1",
    label="Posterior 5-95% (mag)",
)
ax.plot(t_axis[mask_top], model_mag_map[mask_top], "C1-", lw=2, label="MAP (mag)")
ax.plot(t_axis[mask_top], truth_mag_mean[mask_top], "r-", lw=2, label="Reported PSPL")


ax.invert_yaxis()
ax.set_xlim(*x_top)
ax.set_ylim(18.4, 17.0)
ax.set_ylabel("I magnitude")
ax.legend(loc="best", fontsize=8, markerscale=2)
# ax.set_title("KMT-2019-BLG-2073 — Truth in magnitude space")
ax.grid(ls="--", alpha=0.3)

# Bottom panel (zoom)
ax = axes[1]
cut = I[(I["HJDp"] >= x_zoom[0]) & (I["HJDp"] <= x_zoom[1])]
ax.errorbar(
    I["HJDp"],
    I["mag"],
    yerr=I["emag"],
    fmt=".",
    ms=3,
    elinewidth=0.6,
    capsize=0,
    alpha=0.7,
)


mask_zoom = (t_axis >= x_zoom[0]) & (t_axis <= x_zoom[1])
# ax.fill_between(t_axis[mask_zoom], truth_mag_p16[mask_zoom], truth_mag_p84[mask_zoom], alpha=0.2)
ax.fill_between(
    t_axis[mask_zoom],
    model_mag_p5[mask_zoom],
    model_mag_p95[mask_zoom],
    alpha=0.2,
    color="C1",
)
ax.plot(t_axis[mask_zoom], model_mag_map[mask_zoom], "C1-", lw=2)
ax.plot(t_axis[mask_zoom], truth_mag_mean[mask_zoom], "r-", lw=2)

ax.invert_yaxis()
ax.set_xlim(*x_zoom)
ax.set_ylim(18.35, 17.0)
ax.set_xlabel("HJD - 2450000")
ax.set_ylabel("I magnitude")
ax.grid(ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_PATH, "kmt_2019_blg_2073_truth_magnitude_band.pdf"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.0,
)
plt.close(fig)


print("✓ Done. Plots saved to:", FIGURES_PATH)

# 4_eval_ivr_tarp.py
import os, argparse
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange

from config import *
from utils import (
    load_saved_cpu,
    build_posterior_from_saved,
    generate_augmented_observation,
    ensure_dir,
    check_recoverable,
)
from drp import get_tarp_coverage


# --- small helpers ----------------------------------------------------
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
    if tuple(x.shape[-len(event_shape) :]) == tuple(event_shape):
        return x
    if x.dim() >= 2:
        xt = x.transpose(-2, -1).contiguous()
        if tuple(xt.shape[-len(event_shape) :]) == tuple(event_shape):
            return xt
    raise AssertionError(
        f"x trailing shape {tuple(x.shape[-len(event_shape):])} != event_shape {tuple(event_shape)}"
    )


def count_points_in_tE_window(x_event: torch.Tensor, theta: torch.Tensor) -> int:
    norm_t = x_event[:, 0]  # normalized times [-1,1], padding is -2
    valid = norm_t > -1.5
    times = (norm_t[valid] + 1.0) * (TOTAL_DURATION / 2.0)
    t0 = float(theta[0].cpu())
    tE = float(10.0 ** float(theta[2].cpu()))
    lo, hi = t0 - tE / 2, t0 + tE / 2
    return int(((times >= lo) & (times <= hi)).sum().item())


# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", default=LOADED_POSTERIOR_PATH, help="posterior bundle path"
    )
    ap.add_argument(
        "--num-events", type=int, default=5000, help="number of injected cases"
    )
    ap.add_argument(
        "--num-samples", type=int, default=5000, help="posterior samples per case"
    )
    ap.add_argument(
        "--snr-min", type=float, default=5.0, help="SNR threshold for recoverability"
    )
    ap.add_argument(
        "--min-points", type=int, default=10, help="min points in tE window"
    )
    ap.add_argument("--outdir", default=FIGURES_PATH, help="where to write PDFs/CSVs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=bool, default=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dir(args.outdir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading posterior bundle from {args.ckpt} on {device} …")
    saved = load_saved_cpu(args.ckpt)
    prior, posterior = build_posterior_from_saved(saved, device=device)

    # --- sample injected thetas ---------------------------------------
    print(f"Sampling {args.num_events} injected params from prior …")
    # thetas = prior.sample((args.num_events,))
    thetas = PRIOR_TEST.sample((args.num_events,))
    D = thetas.shape[1]

    # --- generate augmented observations ------------------------------
    print("Generating augmented observations …")
    xs_list = []
    for i in trange(args.num_events, desc="Augment", leave=False):
        xs_list.append(generate_augmented_observation(thetas[i], device=device))
    xs_tensor = torch.stack(xs_list, dim=0)  # [N, MAX_NUM_POINTS, 3]

    # --- recoverability mask ------------------------------------------
    print("Computing recoverability mask …")
    keep = np.zeros(args.num_events, dtype=bool)
    for i in trange(args.num_events, desc="Recoverable", leave=False):
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

        cfg = RecoverCfg(min_points=args.min_points, snr_min=args.snr_min)

        is_recoverable, reason = check_recoverable(
            theta, times_np, flux, phot_err=0.1, cfg=cfg
        )
        keep[i] = is_recoverable

    kept_idx = np.where(keep)[0]
    kept = len(kept_idx)
    print(
        f"Recoverable: {kept}/{args.num_events} "
        f"({100.0*kept/max(1,args.num_events):.1f}%)"
    )

    if kept == 0:
        print("No recoverable cases; consider lowering thresholds.")
        return

    # --- posterior sampling on kept samples only --------------------------------
    x0 = xs_tensor[kept_idx[0]]
    posterior.set_default_x(x0)
    event_shape = get_event_shape(posterior) or tuple(x0.shape[-2:])
    print("posterior x_event_shape:", event_shape)

    med = np.zeros((kept, D), dtype=np.float32)
    q16 = np.zeros((kept, D), dtype=np.float32)
    q84 = np.zeros((kept, D), dtype=np.float32)
    truth = thetas[kept_idx].cpu().numpy()
    # for TARP: (n_samples, n_sims, n_dims)
    samples_stack = []
    with torch.no_grad():
        for j, raw_idx in enumerate(trange(len(kept_idx), desc="Sample", leave=False)):
            idx = int(kept_idx[raw_idx])  # map into original arrays
            x_i = coerce_x_to_event_shape(xs_tensor[idx], event_shape)
            samples = posterior.sample(
                (args.num_samples,), x=x_i, show_progress_bars=False
            )
            s_np = samples.detach().cpu().numpy()
            samples_stack.append(s_np)
            med[j] = np.median(s_np, axis=0)
            q16[j] = np.quantile(s_np, 0.16, axis=0)
            q84[j] = np.quantile(s_np, 0.84, axis=0)

    samples_np = np.stack(samples_stack, axis=1)  # (S, N, D)

    # Regime masks (within the kept subset)
    u0_true = truth[:, 1].astype(float)  # linear u0
    rho_true_lin = np.power(
        10.0, truth[:, 3].astype(float)
    )  # convert log10(rho) -> rho

    mask_u0_gt_rho = u0_true > rho_true_lin  # for the u0 panel
    mask_u0_lt_rho = u0_true < rho_true_lin  # for the rho panel
    mask_all = np.ones_like(mask_u0_gt_rho, dtype=bool)

    # === Combined scatter + POSTERIOR-UNCERTAINTY percentile bands =======

    print("Preparing combined 2x3 figure (5 params + TARP) …")

    # Per-event per-parameter posterior quantiles from the full sample stack
    # samples_np has shape (S, N_kept, D)
    q05_all = np.quantile(samples_np, 0.05, axis=0)  # (N_kept, D)
    q16_all = np.quantile(samples_np, 0.16, axis=0)  # (N_kept, D)
    q50_all = np.quantile(samples_np, 0.50, axis=0)  # (N_kept, D)
    q84_all = np.quantile(samples_np, 0.84, axis=0)  # (N_kept, D)
    q95_all = np.quantile(samples_np, 0.95, axis=0)  # (N_kept, D)

    # ---- Compact styling
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    def _binned_posterior_bands(x_inj, q05, q16, q50, q84, q95, bins=10):
        """
        Bin by injected x, then aggregate the per-event posterior quantiles
        with a robust median across events to form shaded bands.

        Returns centers and arrays (y05, y95, y16, y84, y50), each len=bins, NaN where empty.
        """
        x = np.asarray(x_inj)
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
            return None, (None,) * 5

        edges = np.linspace(xmin, xmax, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        y05 = np.full(bins, np.nan)
        y95 = np.full(bins, np.nan)
        y16 = np.full(bins, np.nan)
        y84 = np.full(bins, np.nan)
        y50 = np.full(bins, np.nan)

        for b in range(bins):
            mb = (x >= edges[b]) & (
                x < edges[b + 1] if b < bins - 1 else x <= edges[b + 1]
            )
            if mb.any():
                y05[b] = np.nanmedian(q05[mb])
                y95[b] = np.nanmedian(q95[mb])
                y16[b] = np.nanmedian(q16[mb])
                y84[b] = np.nanmedian(q84[mb])
                y50[b] = np.nanmedian(q50[mb])
        return centers, (y05, y95, y16, y84, y50)

    def draw_panel(ax, t_inj, r_med, q05, q16, q50, q84, q95, name):
        ax.scatter(t_inj, r_med, s=8, alpha=0.15, color="k")
        centers, bands = _binned_posterior_bands(
            t_inj, q05, q16, q50, q84, q95, bins=10
        )
        if centers is not None:
            y05, y95, y16, y84, _y50 = bands
            ok_outer = np.isfinite(y05) & np.isfinite(y95)
            ok_inner = np.isfinite(y16) & np.isfinite(y84)
            if ok_outer.any():
                ax.fill_between(
                    centers[ok_outer],
                    y05[ok_outer],
                    y95[ok_outer],
                    alpha=0.2,
                    color="#0072B2",
                )
            if ok_inner.any():
                ax.fill_between(
                    centers[ok_inner],
                    y16[ok_inner],
                    y84[ok_inner],
                    alpha=0.4,
                    color="#0072B2",
                )
        vmin, vmax = np.nanmin(t_inj), np.nanmax(t_inj)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.7)  # Ideal y=x
        ax.set_xlabel(f"Injected {name}", fontsize=14)
        ax.set_ylabel(f"Recovered {name}", fontsize=14)
        # ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax) #limits?

    # 2×3 layout: top row 3 params, bottom row 2 params + TARP
    mosaic = [
        ["t0", "u0", "ltE"],
        ["lrho", "fs", "tarp"],
    ]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(10, 6), constrained_layout=True)

    # IVR panels (note the masks for u0 and rho; fs uses all)
    label_fs = (
        PARAM_NAMES[4]
        if (isinstance(PARAM_NAMES, (list, tuple)) and len(PARAM_NAMES) > 4)
        else "f_s"
    )
    panels = [
        ("t0", 0, mask_all, PARAM_NAMES[0]),
        ("u0", 1, mask_u0_gt_rho, PARAM_NAMES[1] + " ($u_0>\\rho$)"),
        ("ltE", 2, mask_all, PARAM_NAMES[2]),
        ("lrho", 3, mask_u0_lt_rho, PARAM_NAMES[3] + " ($u_0<\\rho$)"),
        ("fs", 4, mask_all, label_fs),
    ]

    for key, d, m, label in panels:
        ax = axd[key]
        if d >= D:
            ax.set_title(f"{label} — not present in θ (D={D})")
            ax.axis("off")
            continue
        t = truth[m, d]
        r = med[m, d]
        if t.size == 0:
            ax.set_title(f"{label} — no points")
            ax.axis("off")
            continue
        draw_panel(
            ax,
            t,
            r,
            q05_all[m, d],
            q16_all[m, d],
            q50_all[m, d],
            q84_all[m, d],
            q95_all[m, d],
            label,
        )

    # --- TARP coverage (right-bottom tile) --------------------------------
    print("Running TARP coverage (bootstrap={}) …".format(args.bootstrap))
    ecp, alpha = get_tarp_coverage(
        samples_np, truth, bootstrap=args.bootstrap, norm=True
    )

    # ecp shape: (B, len(alpha)) if bootstrap=True; else (len(alpha),)
    if ecp.ndim == 2:  # bootstrapped
        ecp_median = np.median(ecp, axis=0)
        ecp_lo = np.percentile(ecp, 16, axis=0)
        ecp_hi = np.percentile(ecp, 84, axis=0)
        ecp_05 = np.percentile(ecp, 5, axis=0)
        ecp_95 = np.percentile(ecp, 95, axis=0)
    else:
        ecp_median = ecp
        ecp_lo = ecp_hi = ecp
        ecp_05 = ecp_95 = None

    ax_tarp = axd["tarp"]
    ax_tarp.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax_tarp.plot(alpha, ecp_median, lw=2, label="TARP", color="#0072B2")
    ax_tarp.fill_between(alpha, ecp_lo, ecp_hi, alpha=0.4, color="#0072B2")
    if ecp_05 is not None:
        ax_tarp.fill_between(alpha, ecp_05, ecp_95, alpha=0.2, color="#0072B2")
    ax_tarp.set_xlabel("Credibility Level", fontsize=14)
    ax_tarp.set_ylabel("Expected Coverage", fontsize=14)
    ax_tarp.set_xlim(0, 1)
    ax_tarp.set_ylim(0, 1)
    # ax_tarp.grid(True, linestyle=':')
    ax_tarp.legend()
    ax_tarp.set_aspect("equal", "box")
    # ax_tarp.set_title("TARP Coverage")

    # Save ONE combined figure
    combined_out = os.path.join(args.outdir, "ivr_tarp_2x3.pdf")
    fig.savefig(combined_out, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved combined figure: {combined_out}")


if __name__ == "__main__":
    import csv  # needed above for per-param coverage CSV

    main()

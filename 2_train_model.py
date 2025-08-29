# 2_train_model.py
try:
    import comet_ml
except Exception:
    comet_ml = None

import argparse, math, os, torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from tqdm import trange

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
import copy
from dataclasses import asdict


from config import *
from utils import (
    MicrolensingAugmentationDataset,
    ensure_dir,
    set_global_seeds,
    save_posterior_bundle,
    load_saved_cpu,
    RecoverCfg,
)

from model import TransformerEmbeddingNet


# --------------------------- scheduler helper --------------------------- #
def _make_scheduler(optim, t_max=None):
    if LR_SCHEDULER["name"] == "plateau":
        return ReduceLROnPlateau(
            optim,
            mode="min",
            factor=LR_SCHEDULER["factor"],
            patience=LR_SCHEDULER["patience"],
            min_lr=LR_SCHEDULER["min_lr"],
            verbose=True,
        )
    if LR_SCHEDULER["name"] == "step":
        return StepLR(
            optim, step_size=LR_SCHEDULER["step_size"], gamma=LR_SCHEDULER["gamma"]
        )
    if LR_SCHEDULER["name"] == "cosine":
        return CosineAnnealingLR(
            optim, T_max=t_max or MAX_EPOCHS, eta_min=LR_SCHEDULER["min_lr"]
        )
    return None


# ---------------------------- dataloader helper ------------------------ #
def build_dataloaders(dataset, batch_size, val_fraction):
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    NUM_WORKERS = 1  # increase this if you have more cpus available
    common = dict(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
    )


# ----------------------------- training loop --------------------------- #
def train_density_estimator(
    density_estimator,
    train_loader,
    val_loader,
    max_epochs,
    patience,
    lr,
    device,
    experiment,
    on_improve=None,
    lr_embed=None,
    freeze_embed=False,
):
    # --- param groups (only if needed) ---
    split = freeze_embed or (lr_embed is not None)
    if hasattr(density_estimator, "embedding_net") and split:
        embed_params = list(density_estimator.embedding_net.parameters())
        embed_ids = {id(p) for p in embed_params}
        rest_params = [
            p for p in density_estimator.parameters() if id(p) not in embed_ids
        ]

        if freeze_embed:
            for p in embed_params:
                p.requires_grad = False
            param_groups = [{"params": rest_params, "lr": lr}]
        else:
            param_groups = [
                {"params": rest_params, "lr": lr},
                {
                    "params": embed_params,
                    "lr": (lr_embed if lr_embed is not None else lr * 0.25),
                },
            ]
        optim = torch.optim.Adam(param_groups)
    else:
        optim = torch.optim.Adam(density_estimator.parameters(), lr=lr)

    sched = _make_scheduler(optim, t_max=max_epochs)

    best_val, epochs_bad = math.inf, 0
    best_state = copy.deepcopy(density_estimator.state_dict())  # <-- init

    for epoch in range(max_epochs):

        # ---- TRAIN ---------------------------------------------------- #
        density_estimator.train()
        running = 0.0
        for xb, thetab in train_loader:
            xb, thetab = xb.to(device, non_blocking=True), thetab.to(
                device, non_blocking=True
            )
            optim.zero_grad(set_to_none=True)
            loss = density_estimator.loss(thetab, condition=xb).mean()
            if not torch.isfinite(loss):
                print("⚠️  non-finite loss – skip batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(density_estimator.parameters(), 1.0)
            optim.step()
            running += loss.item() * xb.size(0)

        train_NLL = running / len(train_loader.dataset)

        # ---- VALIDATE ------------------------------------------------- #
        density_estimator.eval()
        running = 0.0
        with torch.no_grad():
            for xb, thetab in val_loader:
                xb, thetab = xb.to(device, non_blocking=True), thetab.to(
                    device, non_blocking=True
                )
                running += density_estimator.loss(
                    thetab, condition=xb
                ).mean().item() * xb.size(0)
        val_NLL = running / len(val_loader.dataset)

        # ---- scheduler / logging ------------------------------------- #
        if sched:
            (
                sched.step(val_NLL)
                if isinstance(sched, ReduceLROnPlateau)
                else sched.step()
            )
            experiment.log_metric(
                "learning_rate", optim.param_groups[0]["lr"], step=epoch
            )

        experiment.log_metric("training_loss_epoch", train_NLL, step=epoch)
        experiment.log_metric("validation_loss_epoch", val_NLL, step=epoch)

        if val_NLL < best_val:
            best_val, epochs_bad = val_NLL, 0
            best_state = copy.deepcopy(density_estimator.state_dict())
            if on_improve is not None:
                on_improve(best_state, epoch=epoch + 1, val=best_val)
        else:
            epochs_bad += 1
            if epochs_bad >= patience:
                print(f"Early-stopping after {epoch+1} epochs")
                break

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:4d} | train NLL {train_NLL:.3f} | "
                f"val NLL {val_NLL:.3f} | LR {optim.param_groups[0]['lr']:.2e}"
            )

    density_estimator.load_state_dict(best_state)
    return density_estimator, best_val


# ----------------------------- main ------------------------------------ #
def main():
    # ---- argparse ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comet", type=bool, default=False)
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a previous bundle (.pkl/.pt) to warm-start from",
    )
    ap.add_argument(
        "--freeze-embed",
        action="store_true",
        help="Freeze transformer embedding during fine-tune",
    )
    ap.add_argument(
        "--lr-embed",
        type=float,
        default=None,
        help="Optional lower LR for embedding params (if not frozen)",
    )
    args = ap.parse_args()

    set_global_seeds(args.seed)
    ensure_dir(TRAINED_POSTERIOR_PATH)

    # ---- logging ----
    if args.comet:
        experiment = comet_ml.Experiment(auto_metric_logging=False)
    else:

        class _NoComet:
            def __getattr__(self, name):
                return lambda *a, **k: None

        experiment = _NoComet()

    experiment.log_parameters(
        {
            k: globals()[k]
            for k in [
                "TOTAL_DURATION",
                "MAX_NUM_POINTS",
                "PHOT_ERR_MIN",
                "PHOT_ERR_MAX",
                "AUG_NUM_GAPS_RANGE",
                "AUG_GAP_LENGTH_RANGE",
                "AUG_DROPOUT_RATE_RANGE",
                "NUM_AUGMENTATIONS_PER_SIM",
                "D_MODEL",
                "N_HEAD",
                "D_HID",
                "N_LAYERS",
                "DROPOUT",
            ]
        }
    )
    experiment.log_parameters(
        {
            "BATCH_SIZE": args.batch_size,
            "MAX_EPOCHS": args.max_epochs,
            "PATIENCE": args.patience,
            "LEARNING_RATE": args.lr,
            "SEED": args.seed,
        }
    )

    RECOVER_CFG = RecoverCfg()  # or RecoverCfg(n_peak_min=2, base_k=3.0, ...)
    experiment.log_parameters(
        {f"recover_cfg.{k}": v for k, v in asdict(RECOVER_CFG).items()}
    )

    print(
        {
            "TOTAL_DURATION": TOTAL_DURATION,
            "MAX_NUM_POINTS": MAX_NUM_POINTS,
            "BATCH_SIZE": args.batch_size,
            "MAX_EPOCHS": args.max_epochs,
            "PATIENCE": args.patience,
            "LR": args.lr,
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- load sims ----
    base_data = torch.load(BASE_SIMS_PATH, map_location="cpu")
    base_thetas = base_data["thetas"].cpu()
    experiment.log_parameter("num_base_sims", len(base_thetas))

    dataset = MicrolensingAugmentationDataset(
        thetas=base_thetas,
        n_augs=NUM_AUGMENTATIONS_PER_SIM,
    )
    print("Virtual dataset length:", len(dataset))

    embed = TransformerEmbeddingNet(
        d_model=D_MODEL,
        nhead=N_HEAD,
        d_hid=D_HID,
        nlayers=N_LAYERS,
        dropout=DROPOUT,
        input_dim=3,
    ).to(device)

    builder = posterior_nn(
        model="maf", embedding_net=embed, z_score_x="none", z_score_theta="independent"
    )

    init_x, init_theta = next(iter(DataLoader(dataset, batch_size=64, shuffle=True)))
    flow = builder(init_theta, init_x).to(device)

    if args.resume is not None:
        print(f"[resume] loading bundle from {args.resume}")
        saved = load_saved_cpu(args.resume)
        try:
            sd = (
                saved.get("flow_state_dict", None)
                or saved["density_estimator"].state_dict()
            )
            missing, unexpected = flow.load_state_dict(sd, strict=False)
            print(
                f"[resume] loaded; missing={len(missing)}, unexpected={len(unexpected)}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict from {args.resume}: {e}")

        # Freeze or set LR for embedding if requested
        if args.freeze_embed and hasattr(flow, "embedding_net"):
            for p in flow.embedding_net.parameters():
                p.requires_grad = False
            print("[resume] embedding frozen.")

    # ---- save bundle (robust) ----
    embed_cfg = {
        "D_MODEL": D_MODEL,
        "N_HEAD": N_HEAD,
        "D_HID": D_HID,
        "N_LAYERS": N_LAYERS,
        "DROPOUT": DROPOUT,
        "INPUT_DIM": 3,
    }

    def _save_best(state_dict, epoch, val):
        # load the provided best weights into the current flow and save a bundle
        flow.load_state_dict(state_dict)
        tmp_path = TRAINED_POSTERIOR_PATH + ".tmp"
        save_posterior_bundle(
            flow,
            PRIOR,
            embed_cfg,
            tmp_path,
            meta={
                "checkpoint": True,
                "best_val": float(val),
                "epoch": int(epoch),
                "recover_cfg": asdict(RECOVER_CFG),
            },
        )
        os.replace(tmp_path, TRAINED_POSTERIOR_PATH)  # atomic rename
        print(
            f"[checkpoint] saved best @ epoch {epoch} (val={val:.4f}) → {TRAINED_POSTERIOR_PATH}"
        )

    train_loader, val_loader = build_dataloaders(
        dataset, args.batch_size, VALIDATION_FRACTION
    )

    print("Starting explicit training loop …")
    flow, best_val = train_density_estimator(
        flow,
        train_loader,
        val_loader,
        args.max_epochs,
        args.patience,
        args.lr,
        device,
        experiment,
        on_improve=_save_best,
        lr_embed=args.lr_embed,
        freeze_embed=args.freeze_embed,
    )
    experiment.log_metric("best_validation_loss", best_val)

    save_posterior_bundle(
        flow,
        PRIOR,
        embed_cfg,
        TRAINED_POSTERIOR_PATH,
        meta={
            "checkpoint": False,
            "best_val": float(best_val),
            "train_args": {
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "lr": args.lr,
                "seed": args.seed,
            },
            "recover_cfg": asdict(RECOVER_CFG),
        },
    )

    experiment.end()
    print(f"✓ Saved robust posterior bundle to {TRAINED_POSTERIOR_PATH}")


if __name__ == "__main__":
    main()

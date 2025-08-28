# config.py
import torch
from sbi.utils import BoxUniform
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Simulation Parameters ---
TOTAL_DURATION = 20.0
MIN_CADENCE_DAYS = 0.25 / 24.0 # Minimum cadence in days (15 mins)

NUM_POINTS_BASE = int(TOTAL_DURATION / MIN_CADENCE_DAYS)
MAX_NUM_POINTS = 1000  # Padding length for the networkc
PHOT_ERR_MIN = 0.001  # Min photometric error for augmentation
PHOT_ERR_MAX = 0.02  # Max photometric error for augmentation
BASE_SIM_NOISE = 0.0  # Noise for master simulations should be zero
VALIDATION_FRACTION = 0.2
MAX_TRIES = 5 # Maximum number of tries to generate a valid light curve

AUG_NUM_GAPS_RANGE = (0, 3)         # The number of seasonal gaps.
AUG_GAP_LENGTH_RANGE = (1, 10)      # The length of each gap in days.
AUG_DROPOUT_RATE_RANGE = (0.0, 0.6) # The fraction of points to randomly drop.
NUM_AUGMENTATIONS_PER_SIM = 1


# --- Prior Definition ---
# [t_0, u_0, log10_t_E, log10_rho, blend_fs]
PRIOR_LOW = torch.tensor([
    0.0, #t0
    0.0, #u0
    torch.log10(torch.tensor(0.1)), # log10_tE
    torch.log10(torch.tensor(0.01)), # log10_rho
    0.1  # blend_fs: min value
], device=DEVICE)

PRIOR_HIGH = torch.tensor([
    TOTAL_DURATION, # t0: max duration
    1.5, # u0: max value
    torch.log10(torch.tensor(20.0)), # log10_tE: max value
    torch.log10(torch.tensor(10.0)), # log10_rho: max value
    1.0  # blend_fs: max value
], device=DEVICE)

PRIOR_TEST_LOW = torch.tensor([
    0.0, #t0
    0.0, #u0
    torch.log10(torch.tensor(0.1)), # log10_tE
    torch.log10(torch.tensor(0.01)), # log10_rho
    0.1  # blend_fs: min value
], device=DEVICE)

PRIOR_TEST_HIGH = torch.tensor([
    TOTAL_DURATION, # t0: max duration
    1.0, # u0: max value
    torch.log10(torch.tensor(20.0)), # log10_tE: max value
    torch.log10(torch.tensor(10.0)), # log10_rho: max value
    1.0  # blend_fs: max value
], device=DEVICE)

PRIOR = BoxUniform(low=PRIOR_LOW, high=PRIOR_HIGH)
PRIOR_TEST = BoxUniform(low=PRIOR_TEST_LOW, high=PRIOR_TEST_HIGH)

PARAM_NAMES = [r"$t_0$", r"$u_0$", r"$\log_{10}(t_E)$", r"$\log_{10}(\rho)$", r"$f_s$"]

# --- Network & Training Parameters --
D_MODEL = 256
N_HEAD = 8
D_HID = 512
N_LAYERS = 6
DROPOUT = 0.0
BATCH_SIZE = 128
MAX_EPOCHS = 400
PATIENCE = 30  # Early stopping patience
LEARNING_RATE = 1e-4

LR_SCHEDULER = dict(
    name="plateau",          # "plateau", "step", "cosine", or None
    factor=0.5,              # LR by 0.5
    patience=10,              # wait 10 epochs w/o improvement
    min_lr=1e-6,
    # step-scheduler extras:
    step_size=20,
    gamma=0.1,
)

# --- File Paths ---
#All artifacts live under ./outputs by default; override with env vars if desired.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "outputs"))
BASE_SIMS_PATH = os.environ.get("BASE_SIMS_PATH", os.path.join(OUTPUT_DIR, "base_simulations.pt"))
# TRAINED_POSTERIOR_PATH = os.environ.get("TRAINED_POSTERIOR_PATH", os.path.join(OUTPUT_DIR, "model_best.pkl"))
TRAINED_POSTERIOR_PATH = os.environ.get("TRAINED_POSTERIOR_PATH", "model_trained.pkl")
LOADED_POSTERIOR_PATH = os.environ.get("LOADED_POSTERIOR_PATH", "model_best.pkl")
FIGURES_PATH = os.environ.get("FIGURES_PATH", os.path.join(OUTPUT_DIR, "figures"))
KMT_DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data/pysis"))
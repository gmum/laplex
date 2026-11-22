# ---------------------------------------------------------------------------
# LaplexLinear  |  frozen backbone  |  head-only training
# ---------------------------------------------------------------------------

# --- Backbone ---------------------------------------------------------------
MODEL = "swin_t"          # "swin_t" | "maxvit_t"
FROZEN = True

# --- Head -------------------------------------------------------------------
RANK = 16
AB_SCALE = "auto"         # "auto" or float
C_SCALE  = "auto"         # "auto" or float
TAU_INIT = 1.0
SIGMA_INIT = 0.5

# --- Data -------------------------------------------------------------------
# Update DATA_ROOT and VAL_ROOT to match your local paths before running.
DATA_ROOT = "/data/imagenet"
VAL_ROOT  = "~/imagenet_val_classes"
BATCH_SIZE   = 128
NUM_WORKERS  = 8

# --- Optimiser --------------------------------------------------------------
LR_BASE       = 1e-4       # bias only, backbone is frozen
LR_EXP        = 1e-4       # a, b, log_tau
WEIGHT_DECAY  = 0.0
WD_BASE       = 0.0
WD_EXP        = 0.0
BETAS         = (0.9, 0.999)
EPS           = 1e-8
GRAD_CLIP     = None

# --- Scheduler --------------------------------------------------------------
EPOCHS        = 30
WARMUP_EPOCHS = 0
ETA_MIN       = 1e-6

# --- Loss -------------------------------------------------------------------
LABEL_SMOOTHING = 0.1

# --- Reproducibility --------------------------------------------------------
SEED                    = 0
CUDNN_BENCHMARK         = False
CUDNN_DETERMINISTIC     = True
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

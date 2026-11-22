# ---------------------------------------------------------------------------
# LoRALinear  |  unfrozen backbone  |  full fine-tuning
# ---------------------------------------------------------------------------

# --- Backbone ---------------------------------------------------------------
MODEL = "swin_t"        # "swin_t" | "maxvit_t"
FROZEN = False

# --- Head -------------------------------------------------------------------
RANK       = 16
SIGMA_INIT = 0.5

# --- Data -------------------------------------------------------------------
# Update DATA_ROOT and VAL_ROOT to match your local paths before running.
DATA_ROOT = "/data/imagenet"
VAL_ROOT  = "~/imagenet_val_classes"
BATCH_SIZE   = 128
NUM_WORKERS  = 8

# --- Optimiser --------------------------------------------------------------
LR_BASE       = 1e-5       # backbone + bias
LR_EXP        = 1e-4       # LoRA matrices (A, B)
WEIGHT_DECAY  = 0.05
WD_BASE       = 0.05
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

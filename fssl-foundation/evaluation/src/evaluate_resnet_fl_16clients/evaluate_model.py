# evaluate_model.py

import os
import subprocess
from pathlib import Path

# ── Site-specific configuration (edit or set env vars before running) ----
project_root    = os.environ.get("PROJECT_ROOT")
eval_results    = os.environ.get("EVAL_RESULTS_DIR")
CHECKPOINTS_DIR = os.environ.get(
    "CHECKPOINTS_DIR",
    f"{project_root}/ZODPretraining/src/saved_models/non-IID_split/country_code/16_clients/checkpoints"
)
DATAPATH        = os.environ.get("ZOD_DATA_DIR")

# ─────────────────────────────────────────────────────────────────────────────

MODEL = "fl_16clients"   # "centralized" | "fl_4clients" | "fl_16clients"

CHECKPOINTS = [
    # (label,        checkpoint_path)
    ("epoch_025",   f"{CHECKPOINTS_DIR}/global_round_0005.pth"),
    ("epoch_050",   f"{CHECKPOINTS_DIR}/global_round_0010.pth"),
    ("epoch_075",   f"{CHECKPOINTS_DIR}/global_round_0015.pth"),
    ("epoch_100",   f"{CHECKPOINTS_DIR}/global_round_0020.pth"),
    ("epoch_125",   f"{CHECKPOINTS_DIR}/global_round_0025.pth"),
    ("epoch_150",   f"{CHECKPOINTS_DIR}/global_round_0030.pth"),
    ("epoch_175",   f"{CHECKPOINTS_DIR}/global_round_0035.pth"),
    ("epoch_200",   f"{CHECKPOINTS_DIR}/global_round_0040.pth"),
    ("epoch_225",   f"{CHECKPOINTS_DIR}/global_round_0045.pth"),
    ("epoch_250",   f"{CHECKPOINTS_DIR}/global_round_0050.pth"),
]

# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__)
EVAL_DIR    = CONFIG_PATH.parent
RESULTS_DIR = Path(f"{eval_results}/{MODEL}_FROZEN_BACKBONE")


def update_config(ckpt_path: str, eval_type: str, output_dir: Path):
    CONFIG_PATH.write_text(f"""\
train = True
output_path                = "{output_dir}/fasterrcnn_{eval_type}.pth"
num_epochs                 = 20
eval_interval              = 2  # evaluate and save checkpoint every N epochs
score_threshold            = 0.3
batch_size                 = 2
num_workers                = 64
load_full_fasterrcnn_model = False
model_checkpoint_path      = "{ckpt_path}"
freeze_backbone            = True
eval_output                = "{output_dir}/results.txt"
backbone                   = "resnet_supervised"
datapath                   = "{DATAPATH}"
quick_train_subset         = None
quick_test_subset          = None
""")


def main():
    for label, ckpt_path in CHECKPOINTS:
        print(f"\n{'='*50}")
        print(f"Checkpoint: {label}  ({ckpt_path})")

        coco_dir   = RESULTS_DIR / label 

        # Skip if already done
        if (coco_dir / "results.txt").exists():
            print("  Already done, skipping.")
            continue

        coco_dir.mkdir(parents=True,   exist_ok=True)

        update_config(ckpt_path, "coco", coco_dir)
        
        # Run on GPU 0
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        print("  [GPU 0] Started evaluation with COCO pretrained heads...")
        
        result = subprocess.run(
            ["python", "fasterrcnn_COCO_pretrained.py"],
            cwd=EVAL_DIR,
            env=env
        )

        if result.returncode == 0:
            print("  [GPU 0] Evaluation completed successfully.")
        else:
            print("  [GPU 0] Evaluation failed. Check logs for details.")
            break

    print("\nAll checkpoints evaluated.")


if __name__ == "__main__":
    main()
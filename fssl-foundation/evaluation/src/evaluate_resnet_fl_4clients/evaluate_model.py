# evaluate_model.py

import os
import subprocess
from pathlib import Path

# ── Site-specific configuration (edit or set env vars before running) ----
project_root    = os.environ.get("PROJECT_ROOT")
eval_results    = os.environ.get("EVAL_RESULTS_DIR")
CHECKPOINTS_DIR = os.environ.get(
    "CHECKPOINTS_DIR",
    f"{project_root}/ZODPretraining/src/saved_models/non-IID_split/country_code/4_clients/checkpoints"
)
DATAPATH        = os.environ.get("ZOD_DATA_DIR")

# ─────────────────────────────────────────────────────────────────────────────

MODEL = "fl_4clients"   # "centralized" | "fl_4clients" | "fl_16clients"

CHECKPOINTS = [
    # (label,        checkpoint_path)
    ("epoch_275",   f"{CHECKPOINTS_DIR}/global_round_0055.pth"),
    ("epoch_300",   f"{CHECKPOINTS_DIR}/global_round_0060.pth"),
    ("epoch_325",   f"{CHECKPOINTS_DIR}/global_round_0065.pth"),
    ("epoch_350",   f"{CHECKPOINTS_DIR}/global_round_0070.pth"),
    ("epoch_375",   f"{CHECKPOINTS_DIR}/global_round_0075.pth"),
    ("epoch_400",   f"{CHECKPOINTS_DIR}/global_round_0080.pth"),
    ("epoch_425",   f"{CHECKPOINTS_DIR}/global_round_0085.pth"),
    ("epoch_450",   f"{CHECKPOINTS_DIR}/global_round_0090.pth"),
    ("epoch_475",   f"{CHECKPOINTS_DIR}/global_round_0095.pth"),
    ("epoch_500",   f"{CHECKPOINTS_DIR}/global_round_0100.pth"),
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
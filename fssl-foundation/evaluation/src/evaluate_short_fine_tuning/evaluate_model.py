import os
import subprocess
from pathlib import Path

project_root    = os.environ.get("PROJECT_ROOT")
DATAPATH        = os.environ.get("ZOD_DATA_DIR")

CHECKPOINTS = [
    ("public_DINO", os.environ.get(
        "MODEL_CHECKPOINT_PATH",
        f"{project_root}/ZODPretraining/src/saved_models/public_DINO_resnet50_weights_FIXED.pth"
    )),
]

CONFIG_PATH = Path(__file__)
EVAL_DIR    = CONFIG_PATH.parent
RESULTS_DIR = Path(os.environ.get(
    "SHORT_FT_RESULTS_DIR",
    f"{project_root}/evaluation/src/evaluate_short_fine_tuning/evaluation_results/ablation_study/shorter_fine_tuning/public_DINO_run2"
))


def update_config(ckpt_path: str, label: str, output_dir: Path):
    CONFIG_PATH.write_text(f"""\
train = True
output_path                = "{output_dir}/fasterrcnn_{label}.pth"
num_epochs                 = 20
eval_interval              = 2  # evaluate every N epochs during training
score_threshold            = 0.3
batch_size                 = 2
num_workers                = 64
load_full_fasterrcnn_model = False
model_checkpoint_path      = "{ckpt_path}"
freeze_backbone            = False
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

        output_dir = RESULTS_DIR / label

        if (output_dir / "results.txt").exists():
            print("  Already done, skipping.")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        update_config(ckpt_path, label, output_dir)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        print("  [GPU 0] Started evaluation with COCO pretrained heads...")

        result = subprocess.run(
            ["python", "fasterrcnn_COCO_pretrained.py"],
            cwd=EVAL_DIR,
            env=env,
        )

        if result.returncode == 0:
            print("  [GPU 0] Evaluation completed successfully.")
        else:
            print("  [GPU 0] Evaluation failed. Check logs for details.")
            break

    print("\nAll checkpoints evaluated.")


if __name__ == "__main__":
    main()

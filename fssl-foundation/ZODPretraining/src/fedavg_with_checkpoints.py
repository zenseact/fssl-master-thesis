"""
Minimal FedAvg subclass that saves a numbered checkpoint every N rounds.
Used by both job.py and job_with_sampling.py.
"""

import os

import numpy as np
import torch
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_common.abstract.fl_model import FLModel


class FedAvgWithCheckpoints(FedAvg):
    """
    FedAvg with periodic numbered checkpoint saving.
    Overrides only save_model() — all aggregation, sampling,
    early stopping, etc. are inherited unchanged from FedAvg.
    """

    def __init__(
        self,
        checkpoint_interval: int = 5,
        checkpoint_dir: str = os.environ.get(
            "CHECKPOINT_DIR"),
        *args,
        **kwargs,
    ):
        """
        Args:
            checkpoint_interval: save checkpoint every this many rounds.
            checkpoint_dir: absolute path where checkpoints are saved.
            *args, **kwargs: passed directly to FedAvg.__init__
        """
        super().__init__(*args, **kwargs)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir

    def save_model(self, model: FLModel) -> None:
        # self.current_round is set by FedAvg.run() before calling save_model
        round_num = self.current_round + 1  # convert to 1-indexed

        if round_num % self.checkpoint_interval == 0:
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                dst = os.path.join(
                    self.checkpoint_dir,
                    f"global_round_{round_num:04d}.pth",
                )
                converted = {}
                for k, v in model.params.items():
                    if isinstance(v, np.ndarray):
                        converted[k] = torch.from_numpy(v)
                    elif isinstance(v, (np.floating, np.integer)):
                        converted[k] = torch.tensor(v.item())
                    else:
                        converted[k] = v
                torch.save(converted, dst)
                self.info(f"Checkpoint saved: round {round_num} → {dst}")
            except Exception as e:
                self.warning(f"Checkpoint save failed at round {round_num}: {e}")

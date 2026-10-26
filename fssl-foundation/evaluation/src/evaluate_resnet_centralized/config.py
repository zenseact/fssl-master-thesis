import os

train = True
project_root    = os.environ.get("PROJECT_ROOT")
zod_data_dir    = os.environ.get("ZOD_DATA_DIR")
eval_results    = os.environ.get("EVAL_RESULTS_DIR")
output_path                = f"{eval_results}/centralized_FROZEN_BACKBONE/epoch_500/fasterrcnn_coco.pth"
num_epochs                 = 20
eval_interval              = 2
score_threshold            = 0.3
batch_size                 = 2
num_workers                = 64
load_full_fasterrcnn_model = False
model_checkpoint_path      = f"{project_root}/ZODPretraining/src/saved_models/centralized_checkpoints/500_epochs/checkpoint0500.pth"
freeze_backbone            = True
eval_output                = f"{eval_results}/centralized_FROZEN_BACKBONE/epoch_500/results.txt"
backbone                   = "resnet_supervised"
datapath                   = zod_data_dir
quick_train_subset         = None
quick_test_subset          = None

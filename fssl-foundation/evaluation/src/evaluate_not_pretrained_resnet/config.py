import os

train = True
project_root    = os.environ.get("PROJECT_ROOT")
zod_data_dir    = os.environ.get("ZOD_DATA_DIR")
eval_results    = os.environ.get("EVAL_RESULTS_DIR")
output_path                = f"{eval_results}/ablation_study/shorter_fine_tuning/public_DINO_run3/fasterrcnn.pth"
num_epochs                 = 20
eval_interval              = 2
score_threshold            = 0.3
batch_size                 = 2
num_workers                = 64
load_full_fasterrcnn_model = False
model_checkpoint_path      = f"{project_root}/ZODPretraining/src/saved_models/public_DINO_resnet50_weights_FIXED.pth"
freeze_backbone            = False
eval_output                = f"{eval_results}/ablation_study/shorter_fine_tuning/public_DINO_run3/evaluation_results.txt"
backbone                   = "resnet_supervised"
datapath                   = zod_data_dir
quick_train_subset         = None
quick_test_subset          = None

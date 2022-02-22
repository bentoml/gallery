# transformers #
# ------------ #
TASKS = "text-classification"
MODEL = "j-hartmann/emotion-english-distilroberta-base"

# BentoML #
# ------- #
BENTOML_PRETRAINED_RUNNER_NAME = "roberta_default_runner"
BENTOML_TRANSFER_RUNNER_NAME = "roberta_transfer_runner"
BENTOML_FINETUNE_NAME = "roberta_fine_tune"

# training parameters #
# ------------------- #
NUM_LABELS = 6
NUM_EPOCHS = 1
NUM_EXAMPLES = 400
BATCH_SIZE = 64
LR = 2e-5
WDECAY = 0.01

# Contributed by {{cookiecutter.author}}.
import argparse
import os
import random

import bentoml
import numpy as np
# import necessary library for training here

K_FOLDS = 5
NUM_EPOCHS = 3
LOSS_FUNCTION = ...  # should be a callable function here


# reproducible setup for testing
# this usually update seeds
seed = 42
random.seed(seed)
np.random.seed(seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BentoML {{cookiecutter.framework}} {{cookiecutter.project_name}} Example')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, metavar='N',
                        help=f'number of epochs to train (default: {NUM_EPOCHS})')
    parser.add_argument('--k-folds', type=int, default=K_FOLDS, metavar='N',
                        help=f'number of folds for k-fold cross-validation (default: {K_FOLDS}, 1 to disable cv)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enable CUDA training')
    parser.add_argument('--model-name', type=str, default="{{ cookiecutter.__project_slug }}",
                        help='name for saved the model')

    args = parser.parse_args()

    trained_model = ...

    bentoml.{{ cookiecutter.__project_dir }}.save(
        args.model_name,
        trained_model,
        metadata=metadata,
    )

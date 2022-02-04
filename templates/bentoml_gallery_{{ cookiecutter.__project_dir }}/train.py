# Contributed by {{cookiecutter.author}}.
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

    bentoml.{{ cookiecutter.__project_dir }}.save(
        "{{ cookiecutter.__project_slug }}",
        trained_model,
        metadata=metadata,
    )

import argparse
import os
import random

import bentoml
import numpy as np
# import necessary library for training here

from model import {{ cookiecutter.project_name.replace(' ', '') }}

K_FOLDS = 5
NUM_EPOCHS = 3
LOSS_FUNCTION = ...  # should be a callable function here


# reproducible setup for testing
# this usually update seeds
seed = 42
random.seed(seed)
np.random.seed(seed)

def _dataloader_init_fn(worker_id):
    np.random.seed(seed)


def get_dataset(): ...


def train_epoch(model, optimizer, loss_function, train_loader, epoch, device="cpu"): ...


def test_model(model, test_loader, device="cpu"): ...


def cross_validate(dataset, epochs=NUM_EPOCHS, k_folds=K_FOLDS): ...


def train(dataset, epochs=NUM_EPOCHS, device="cpu"): ...


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
    use_cuda = args.cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_set, test_set = get_dataset()
    test_loader = ...

    if args.k_folds > 1:
        cv_results = cross_validate(train_set, args.epochs, args.k_folds)
    else:
        cv_results = {}

    trained_model = train(train_set, args.epochs, device)
    correct, total = test_model(trained_model, test_loader, device)

    # training related 
    metadata = {
        "acc": float(correct)/total,
        "cv_stats": cv_results,
    }

    bentoml.{{ cookiecutter.__project_dir }}.save(
        args.model_name,
        trained_model,
        metadata=metadata,
    )

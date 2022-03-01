# Text Summarization Tutorial

This is a sample project demonstrating basic usage of BentoML, the machine learning model serving library.

In this project, we will train a summarization model using PyTorch on the [Reddit TL;DR](https://zenodo.org/record/1043504) dataset, build an ML service for the model, serve the model behind an HTTP endpoint, and containerize the model server as a docker image for production deployment.

This project is also available to run from a notebook: https://github.com/bentoml/gallery/blob/main/pytorch_seq2seq/pytorch_seq2seq.ipynb

### Install Dependencies

Install python packages required for running this project:

```bash
pip install -r ./requirements.txt
```

### Model Training

First step, train a summarization model with PyTorch:

```bash
python train.py
```

There should now be two new models in the BentoML local model store:

```bash
bentoml models list

> pytorch_seq2seq_encoder
> pytorch_seq2seq_decoder
```

If you'd like to skip training, running the block below will use the pre-trained model for the bentoml runners.

Verify that the model can be loaded as runner from Python shell:

```python
import bentoml
from model import EncoderRNN, DecoderRNN, AttnDecoderRNN, device

MAX_LENGTH = 260

try:
  runner_encoder = bentoml.pytorch.load_runner("pytorch_seq2seq_encoder:latest")
  runner_decoder = bentoml.pytorch.load_runner("pytorch_seq2seq_decoder:latest")
except bentoml.exceptions.NotFound:
  import torch
  seq2seq = torch.load("models/seq2seq.pt") # Load the model
  bentoml.pytorch.save(name="pytorch_seq2seq_encoder", model=seq2seq["encoder"])
  bentoml.pytorch.save(name="pytorch_seq2seq_decoder", model=seq2seq["decoder"])
  runner_encoder = bentoml.pytorch.load_runner("pytorch_seq2seq_encoder:latest")
  runner_decoder = bentoml.pytorch.load_runner("pytorch_seq2seq_decoder:latest")

encoded_sentence = runner_encoder.run("some text to summarize")
print(runner_decoder.run(encoded_sentence))
```

## Host the model with BentoML

Start an API server locally to test the service code above:

```bash
bentoml serve service.py:svc --reload
```

With the `--reload` flag, the API server will automatically restart when the source
file `service.py` is being edited, to boost your development productivity.

Users can also access 127.0.0.1:5000 to access the swaggers docs and interact with the service in real time

### Build Bento for deployment

A `bentofile` is already created in this directory for building a
Bento for the service:

```yaml
service: "seq2seq:svc"
description: "file: ./README.md"
labels:
  owner: bentoml-team
  stage: demo
include:
  - "*.py"
exclude:
  - "tests/"
python:
  packages:
    - torch
```

Note that we exclude `tests/` from the bento using `exclude`.

Simply run `bentoml build` from current directory to build a Bento with the latest
version of the `pytorch_seq2seq` model. This may take a while when running for the first
time for BentoML to resolve all dependency versions:

```
> bentoml build

[01:14:04 AM] INFO     Building BentoML service "pytorch_seq2seq:bmygukdtzpy6zlc5vcqvsoywq" from build context
                       "/home/chef/workspace/gallery/pytorch"
              INFO     Packing model "pytorch_seq2seq:xm6jsddtu3y6zluuvcqvsoywq" from
                       "/home/chef/bentoml/models/pytorch_seq2seq/xm6jsddtu3y6zluuvcqvsoywq"
              INFO     Locking PyPI package versions..
[01:14:05 AM] INFO
                       ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
                       ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
                       ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
                       ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
                       ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
                       ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

              INFO     Successfully built Bento(tag="pytorch_seq2seq:bmygukdtzpy6zlc5vcqvsoywq") at
                       "/home/chef/bentoml/bentos/pytorch_seq2seq/bmygukdtzpy6zlc5vcqvsoywq/"
```

This Bento can now be loaded for serving:

```bash
bentoml serve pytorch_seq2seq:latest --production
```

The Bento directory contains all code, files, models and configs required for running this service.
BentoML standardizes this file structure which enables serving runtimes and deployment tools to be
built on top of it. By default, Bentos are managed under the `~/bentoml/bentos` directory:

```
> cd ~/bentoml/bentos/pytorch_seq2seq && cd $(cat latest)

> tree
.
├── apis
│   └── openapi.yaml
├── bento.yaml
├── env
│   ├── conda
│   ├── docker
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   └── init.sh
│   └── python
│       ├── requirements.lock.txt
│       ├── requirements.txt
│       └── version.txt
├── models
│   └── pytorch_seq2seq
│       ├── eqxdigtybch6nkfb
│       │   ├── model.yaml
│       │   └── saved_model.pt
│       └── latest
├── README.md
└── src
    ├── model.py
    ├── service.py
    └── train.py

9 directories, 15 files
```

### Containerize Bento for deployment

Make sure you have docker installed and docker deamon running, and the following command
will use your local docker environment to build a new docker image, containing the model
server configured from this Bento:

```bash
bentoml containerize pytorch_seq2seq:latest
```

Test out the docker image built:

```bash
docker run -p 5000:5000 pytorch_seq2seq:invwzzsw7li6zckb2ie5eubhd
```

### Sources

1. Reddit TL;DR dataset - https://zenodo.org/record/1043504
2. PyTorch Sequence to Sequence Model - https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

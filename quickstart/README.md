# BentoML Quickstart

This is a sample project demonstrating basic usage of BentoML, the machine learning model serving library.

In this project, we will train a classifier model using Scikit-learn and the Iris dataset, build
an ML service for the model, serve the model behind an HTTP endpoint, and containerize the model
server as a docker image for production deployment.

This project is also available to run from a notebook: https://github.com/bentoml/gallery/blob/main/quickstart/iris_classifier.ipynb

### Install Dependencies

Install python packages required for running this project:
```bash
pip install -r ./requirements.txt
```

### Model Training

First step, train a classification model with sklearn's built-in iris dataset and save the model
with BentoML:

```bash
python train.py
```

This should save a new model in the BentoML local model store:

```bash
bentoml models list
```

Verify that the model can be loaded as runner from Python shell:

```python
import bentoml

runner = bentoml.sklearn.load_runner("iris_clf:latest")

runner.run([5.9, 3. , 5.1, 1.8])  # => array(2)
```

### Create ML Service

The ML Service code is defined in the `iris_classifier.py` file:

```python
# iris_classifier.py
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray


iris_clf_runner = bentoml.sklearn.load_runner("iris_clf:latest")

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    return iris_clf_runner.run(input_series)
```

Start an API server locally to test the service code above:

```bash
bentoml serve iris_classifier:svc --reload
```

With the `--reload` flag, the API server will automatically restart when the source
file `iris_classifier.py` is being edited, to boost your development productivity.


Verify the endpoint can be accessed locally:
```bash
curl -X POST -H "content-type: application/json" --data "[5, 4, 3, 2]" http://127.0.0.1:5000/classify
```


### Build Bento for deployment

A `bentofile` is already created in this directory for building a Bento for the iris_classifier
service:

```yaml
service: "iris_classifier:svc"
description: "file: ./README.md"
labels:
  owner: bentoml-team
  stage: demo
include:
- "*.py"
python:
  packages:
    - scikit-learn
    - pandas
```

Simply run `bentoml build` from current directory to build a Bento with the latest
version of the `iris_clf` model. This may take a while when running for the first
time for BentoML to resolve all dependency versions:

```
> bentoml build

[12/06/2021 17:09:01] INFO     Building BentoML service "iris_classifier:invwzzsw7li6zckb2ie5eubhd" from build context
                               "/Users/chef/workspace/gallery/quickstart"
                      INFO     Packing required models: "iris_clf:7kr6hkcw6ti6zeox2ie5eubhd"
                      INFO     Locking PyPI package versions..
[12/06/2021 17:10:55] INFO     Bento build success, Bento(tag="iris_classifier:invwzzsw7li6zckb2ie5eubhd",
                               path="/Users/chef/bentoml/bentos/iris_classifier/invwzzsw7li6zckb2ie5eubhd/") created
```

This Bento can now be loaded for serving:

```bash
bentoml serve iris_classifier:latest --production
```

The Bento directory contains all code, files, models and configs required for running this service.
BentoML standarlizes this file structure which enables serving runtimes and deployment tools to be
built on top of it. By default, Bentos are managed under the `~/bentoml/bentos` directory:

```
> cd ~/bentoml/bentos/iris_classifier && cd $(cat latest)

> tree
.
├── README.md
├── apis
│   └── openapi.yaml
├── bento.yaml
├── env
│   ├── docker
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   └── init.sh
│   └── python
│       ├── requirements.lock.txt
│       ├── requirements.txt
│       └── version.txt
├── models
│   └── iris_clf
│       ├── latest
│       └── y3f4ijcxj3i6zo6x2ie5eubhd
│           ├── model.yaml
│           └── saved_model.pkl
└── src
    ├── iris_classifier.py
    └── train.py

8 directories, 14 files
```


### Containerize Bento for deployment

Make sure you have docker installed and docker deamon running, and the following command
will use your local docker environment to build a new docker image, containing the model
server configured from this Bento:

```bash
bentoml containerize iris_classifier:latest
```

Test out the docker image built:
```bash
docker run -p 5000:5000 iris_classifier:invwzzsw7li6zckb2ie5eubhd 
```

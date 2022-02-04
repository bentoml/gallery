# BentoML's Sentiment Analysis Tutorial

This is a sample project demonstrating basic usage of BentoML, The Unified Model
Serving Framework.

In this project, There are two parts:

1. We will import a pretrained Roberta model
2. We will also fine-tuning the model and compare the results between the two models.

We then create an ML service for both models, serve it behind an HTTP endpoint, and containerize the model
server as a docker image for production deployment.

We will also enable GPU supports for this service, and deploy it to [Yatai](https://github.com/bentoml/Yatai)

This project is also available to run from a [Jupyter Notebook](https://github.com/bentoml/gallery/blob/main/transformers/roberta_text_classification/transformers_roberta_text_classification_demo.ipynb). You can also try it out on [Colab](https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transformers_roberta_text_classification_demo.ipynb).

### Specification

model: [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english)
framework: [`transformers`](https://huggingface.co/docs/transformers/index)

### Install Dependencies

Install python packages required for running this project:
```bash
pip install -r ./requirements.txt
```

### Model Training

First step, train the Roberta Text Classification with Transformers and then save the model to BentoML local model store:

```bash
python train.py
```

One can check the newly saved model in BentoML local model store:

```bash
bentoml models list
```

Verify that the model can be loaded as runner from an interactive Python shell:

```python
import bentoml
# import necessary library for preprocess

runner = bentoml.transformers.load_runner("roberta_text_classification:latest")

# preprocess an input called `inp`
...

runner.run(inp)
```

### Create ML Service

The ML Service code is defined in the [`service.py`](./service.py) file:

```python
```

We defined the following API for our endpoints with a single runner: [[endpoints]]

Start an API server locally to test the service code above:

```bash
bentoml serve service:svc --reload
```

With the `--reload` flag, the API server will automatically restart when the source
file `service.py` is being edited, to boost your development productivity.


Verify the endpoint can be accessed locally via `curl`:
```bash
curl -H "Content-Type: multipart/form-data" -F'fileobj=@samples/1.png;type=image/png' http://127.0.0.1:5000/[[an_endpoint]]
```


### Build Bento for deployment

A [`bentofile`](./bentofile.yaml) is already created in this directory for building a
Bento for the service:

```yaml
service: "service:svc"
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
    - scikit-learn
    - Pillow

```

Note that we exclude `tests/` from the bento using `exclude`.

Simply run `bentoml build` from current directory to build a Bento with the latest
version of the `transformers_roberta_text_classification` model. This may take a while when running for the first
time for BentoML to resolve all dependency versions:

```
> bentoml build

[01:14:04 AM] INFO     Building BentoML service "transformers_roberta_text_classification:[[bento_tag]]" from build context      
                       "/home/chef/workspace/gallery/pytorch"                                                         
              INFO     Packing model "roberta_text_classification:[[model_tag]]" from                               
                       "/home/chef/bentoml/models/roberta_text_classification/[[model_tag]]"                       
              INFO     Locking PyPI package versions..                                                                 
[01:14:05 AM] INFO                                                                                                     
                       ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░                                   
                       ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░                                   
                       ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░                                   
                       ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░                                   
                       ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗                                   
                       ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝                                   
                                                                                                                       
              INFO     Successfully built Bento(tag="transformers_roberta_text_classification:[[bento_tag]]") at                 
                       "/home/chef/bentoml/bentos/transformers_roberta_text_classification/[[bento_tag]]/"                      
```

This Bento can now be loaded for serving:

```bash
bentoml serve transformers_roberta_text_classification:latest --production
```

The Bento directory contains all code, files, models and configs required for running this service.
BentoML standarlizes this file structure which enables serving runtimes and deployment tools to be
built on top of it. By default, Bentos are managed under the `~/bentoml/bentos` directory:

```
> cd ~/bentoml/bentos/transformers_roberta_text_classification && cd $(cat latest)

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
│   └── transformers_roberta_text_classification
│       ├── [[model_tag]] 
│       │   ├── [[model_content]]
│       │   └── [[model_content]]
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
bentoml containerize transformers_roberta_text_classification:latest
```

Test out the docker image built:
```bash
docker run -p 5000:5000 transformers_roberta_text_classification:[[docker_tag]]
```

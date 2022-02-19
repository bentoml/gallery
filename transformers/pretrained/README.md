## Pretrained Roberta with Transformers :handshake: BentoML

<div align='center'>
    <p align='center'>
        <a href="https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
        <a href="https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="nbviewer"/></a>
        <a href="https://github.com/bentoml/gallery/tree/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb"><img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter" alt="Made with Jupyter"/></a>
    </p>
</div>

We are going to use [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) and serve it under `/predict`


### Instruction

First, import the aboved model to your local BentoML modelstore:
```python
python import_model.py
```

This will create a BentoML Model format with tag `roberta_text_classification`.
One can retrieve this model with:
```bash
bentoml models list
```

Verify this model in an IPython shell:
```python
import bentoml

runner = bentoml.transformers.load_runner("roberta_text_classification",
                                          tasks='text-classification',
                                          return_all_scores=True)

runner.run_batch(["Hello World", "I love you", "I hate you"])
```

#### Create a BentoML service

```python
# service.py
import ast
import typing as t
import logging

import bentoml
from bentoml.io import JSON
from bentoml.io import Text

MODEL_NAME = "roberta_text_classification"
TASKS = "text-classification"

logger = logging.getLogger("bentoml")

clf_runner = bentoml.transformers.load_runner(MODEL_NAME, tasks=TASKS)

all_runner = bentoml.transformers.load_runner(
    MODEL_NAME, name="all_score_runner", tasks=TASKS, return_all_scores=True
)

svc = bentoml.Service(name="pretrained_clf", runners=[clf_runner, all_runner])


def convert_result(res) -> t.Dict[str, t.Any]:
    if isinstance(res, list):
        return {l["label"]: l["score"] for l in res}
    return {res["label"]: res["score"]}


def preprocess(sentence: str) -> t.List[str]:
    return ast.literal_eval(sentence)


def postprocess(
    input_str: t.List[str], output: t.List[t.Dict[str, t.Any]]
) -> t.Dict[str, t.Any]:
    res = {}
    for i, (sent, pred) in enumerate(zip(input_str, output)):
        res[i] = {"inputs": sent, **convert_result(pred)}
        logger.debug(f"entry: {res[i]}")
    return res


@svc.api(input=Text(), output=JSON())
async def sentiment(sentence: str) -> t.Dict[str, t.Any]:
    processed = preprocess(sentence)
    output = await clf_runner.async_run_batch(processed)
    return postprocess(processed, output)


@svc.api(input=Text(), output=JSON())
async def all_scores(sentence: str) -> t.Dict[str, t.Any]:
    processed = preprocess(sentence)
    output = await all_runner.async_run_batch(processed)
    return postprocess(processed, output)
```

We defined two separate endpoints `/sentiment` and `all_scores`, which both
returns a list of predicted emotions from given sentence

Start a service with reload enabled:
```python
bentoml serve service:svc --reload
```
With the `--reload` flag, the API server will automatically restart when the source file `service.py` is being updated.

One can then navigate to `127.0.0.1:3000` and interact with Swagger UI.
One can also verify the endpoints locally with `curl`:
```bash
curl -X POST "http://127.0.0.1:3000/sentiment" \
    -H "accept: application/json" -H "Content-Type: text/plain" \
    -d "[\"I love you\",\"I hope that we will meet one day, but now, our path diverges\"]"
```

We can also do a simple local benchmark with [locust](https://locust.io/):
```bash
locust --headless -u 100 -r 1000 --run-time 2m --host http://127.0.0.1:3000
```

#### Build a Bento for deployment

A `benofile.yaml` can be created to create a Bento with `bentoml build` in the current directory:
```yaml
service: "service:svc"
description: "file: ./README.md"
labels:
  owner: bentoml-team
  stage: demo
include:
- "*.py"
exclude:
- "locustfile.py"
- "tests/"
- "*.ipynb"
python:
  packages:
    - git+https://github.com/huggingface/transformers
    - datasets
```

```bash
» bentoml build
[14:01:58] WARNING  [boot] /Users/aarnphm/mambaforge/lib/python3.9/site-packages/jax/_src/lib/__init__.py:32: UserWarning: JAX on Mac ARM machines is experimental and minimally tested. Please see https://github.com/google/jax/issues/5501 in the event of problems.
                      warnings.warn("JAX on Mac ARM machines is experimental and minimally tested. "

           INFO     [boot] Building BentoML service "pretrained_clf:nhlf7surw2jwlgxi" from build context "/Users/aarnphm/Documents/cs/github/gallery/transformers/pretrained"
           INFO     [boot] Packing model "roberta_text_classification:nzy7ckerl27wrgxi" from "/Users/aarnphm/bentoml/models/roberta_text_classification/nzy7ckerl27wrgxi"
           INFO     [boot] Locking PyPI package versions..
[14:02:20] INFO     [boot]
                    ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
                    ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
                    ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
                    ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
                    ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
                    ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

           INFO     [boot] Successfully built Bento(tag="pretrained_clf:nhlf7surw2jwlgxi") at "/Users/aarnphm/bentoml/bentos/pretrained_clf/nhlf7surw2jwlgxi/"
```

This Bento now can be served with `--production`:
```bash
bentoml serve pretrained_clf:latest --production
```

#### Containerize a Bento

Make sure Docker and daemon is running, then `bentoml containerize` will build
a docker image for the model server aboved:
```bash
bentoml containerize pretrained_clf:latest
```

Test out the newly built docker image:
```bash
docker run -p 3000:3000 
```

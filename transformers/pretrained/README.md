## Pretrained Roberta with Transformers :handshake: BentoML

<!--
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/pretrained/pretrained_roberta.ipynb)[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/pretrained/pretrained_roberta.ipynb)[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://github.com/bentoml/gallery/tree/main/transformers/pretrained/pretrained_roberta.ipynb)
-->
<div align='center'>
    <p align='center'>
        <a href="https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/pretrained/pretrained_roberta.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="nbviewer"/></a>
        <a href="https://github.com/bentoml/gallery/tree/main/transformers/pretrained/pretrained_roberta.ipynb"><img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter" alt="Made with Jupyter"/></a>
    </p>
</div>

We are going to use [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) and serve it with BentoML.


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
import re
import typing as t
import asyncio
import unicodedata

import bentoml
from bentoml.io import JSON
from bentoml.io import Text


MODEL_NAME = "roberta_text_classification"
TASKS = "text-classification"

clf_runner = bentoml.transformers.load_runner(MODEL_NAME, tasks=TASKS)

all_runner = bentoml.transformers.load_runner(
    MODEL_NAME, name="all_score_runner", tasks=TASKS, return_all_scores=True
)

svc = bentoml.Service(name="pretrained_clf", runners=[clf_runner, all_runner])


def normalize(s: str) -> str:
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", s.lower().strip())
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def preprocess(sentence: t.Dict[str, t.List[str]]) -> t.Dict[str, t.List[str]]:
    assert 'text' in sentence, "Given JSON does not contain `text` field"
    if not isinstance(sentence['text'], list):
        sentence['text'] = [sentence['text']]
    return {k: [normalize(s) for s in v] for k, v in sentence.items()}


def convert_result(res) -> t.Dict[str, t.Any]:
    if isinstance(res, list):
        return {l["label"]: l["score"] for l in res}
    return {res["label"]: res["score"]}


def postprocess(
    inputs: t.Dict[str, t.List[str]], outputs: t.List[t.Dict[str, t.Any]]
) -> t.Dict[int, t.Dict[str, t.Union[str, float]]]:
    return {
        i: {"input": sent, **convert_result(pred)}
        for i, (sent, pred) in enumerate(zip(inputs["text"], outputs))
    }


@svc.api(input=Text(), output=JSON())
async def sentiment(sentence: str) -> t.Dict[str, t.Any]:
    res = await clf_runner.async_run(sentence)
    return {"input": sentence, "label": res['label']}


@svc.api(input=JSON(), output=JSON())
async def batch_sentiment(sentences: t.Dict[str, t.List[str]]) -> t.Dict[int, t.Dict[str, t.Union[str, float]]]:
    processed = preprocess(sentences)
    outputs = await asyncio.gather(*[clf_runner.async_run(s) for s in processed["text"]])
    return postprocess(processed, outputs)  # type: ignore


@svc.api(input=JSON(), output=JSON())
async def batch_all_scores(sentences: t.Dict[str, t.List[str]]) -> t.Dict[int, t.Dict[str, t.Union[str, float]]]:
    processed = preprocess(sentences)
    outputs = await asyncio.gather(*[all_runner.async_run(s) for s in processed["text"]])
    return postprocess(processed, outputs)  # type: ignore
```

We defined two separate endpoints `/batch_sentiment` and `batch_all_scores` which creates an inference graph to make use of BentoML's dynamic batching, which both
returns a list of predicted emotions from given sentence. We also create
`/sentiment` endpoints which accept a single sentence as input.

Start a service with reload enabled:
```python
bentoml serve service:svc --reload
```
With the `--reload` flag, the API server will automatically restart when the source file `service.py` is being updated.

One can then navigate to `127.0.0.1:3000` and interact with Swagger UI.
One can also verify the endpoints locally with `curl`:
```bash
curl -X POST "http://127.0.0.1:3000/batch_sentiment" \
    -H "accept: application/json" -H "Content-Type: text/plain" \
    -d "[\"I love you\",\"I hope that we will meet one day, but now, our path diverges\"]"
```

We can also do a simple local benchmark with [locust](https://locust.io/):
```bash
locust --headless -u 100 -r 1000 --run-time 2m --host http://127.0.0.1:3000
```

#### Build a Bento for deployment

A `bentofile.yaml` can be created to create a Bento with `bentoml build` in the current directory:
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
  lock_packages: false
  packages:
    - -f https://download.pytorch.org/whl/cpu/torch_stable.html
    - torch==1.10.2+cpu
    - git+https://github.com/huggingface/transformers
    - datasets
    - pydantic 
```

```bash
» bentoml build
```

```bash
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
» bentoml containerize pretrained_clf:latest
# results image tag: pretrained_clf:zt4vvsurw63thgxi
```

Test out the newly built docker image:
```bash
docker run -p 3000:3000 pretrained_clf:zt4vvsurw63thgxi
```

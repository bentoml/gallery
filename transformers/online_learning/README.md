## Online learning with Transformers :handshake: BentoML

<div align='center'>
    <p align='center'>
        <a href="https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/online_learning/online_learning_roberta.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="nbviewer"/></a>
        <a href="https://github.com/bentoml/gallery/tree/main/transformers/online_learning/online_learning_roberta.ipynb"><img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter" alt="Made with Jupyter"/></a>
    </p>
</div>

We are going to fine tune a version of [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) and serve it with BentoML.


### Instruction

- One can fine tune the model by running [fine-tune notebook](./fine_tune_roberta.ipynb).

- If one prefer to import the fine-tune version, we also provided the [checkpoints](https://huggingface.co/aarnphm/finetune_emotion_distilroberta) for easier access.
```python
FINETUNE_MODEL = "aarnphm/finetune_emotion_distilroberta"
m1 = transformers.AutoModelForSequenceClassification.from_pretrained(FINETUNE_MODEL)
t1 = transformers.AutoTokenizer.from_pretrained(FINETUNE_MODEL)
_ = bentoml.transformers.save("drobert_ft", m1, tokenizer=t1)

```

Make sure to also saved a version of the pretrained model to BentoML modelstore:

```python
PRETRAINED_MODEL = "j-hartmann/emotion-english-distilroberta-base"
m2 = transformers.AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
t2 = transformers.AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
_ = bentoml.transformers.save("emotion_distilroberta_base", m2, tokenizer=t2)
```

#### Create a BentoML service

```python
# service.py
import re
import typing as t
import asyncio
import unicodedata
from pydantic import BaseModel

import bentoml
from bentoml.io import JSON
from bentoml.io import Text

FT_MODEL_TAG = "drobert_ft"
PRETRAINED_MODEL_TAG = "emotion_distilroberta_base"
TASKS = "text-classification"

ft_runner = bentoml.transformers.load_runner(FT_MODEL_TAG, tasks=TASKS, return_all_scores=True)

pretrained_runner= bentoml.transformers.load_runner(PRETRAINED_MODEL_TAG, tasks=TASKS, return_all_scores=True)

svc = bentoml.Service(name="online_learning_ft", runners=[ft_runner, pretrained_runner])

class Prediction(BaseModel):
    input: str
    sadness: float
    joy: float
    love: float
    anger: float
    fear: float
    surprise: float

class Outputs(BaseModel):
    drobert_ft: Prediction
    emotion_distilroberta_base: Prediction

def normalize(s: str) -> str:
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", s.lower().strip())
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def convert_result(res) -> t.Dict[str, t.Any]:
    if isinstance(res, list):
        return {l["label"]: l["score"] for l in res}
    return {res["label"]: res["score"]}


@svc.api(input=Text(), output=JSON(pydantic_model=Outputs))
async def compare(sentences: str) -> t.Dict[str, t.Dict[str, t.Union[str, float]]]:
    processed = normalize(sentences)
    outputs = await asyncio.gather(
        ft_runner.async_run(processed),
        pretrained_runner.async_run(processed)
    )
    return {
        name: {**convert_result(pred)}
        for name, pred in zip(svc.runners.keys(), outputs)
    }

@svc.api(input=Text(), output=Text())
async def online_learning(sentence: str) -> str:...
```

We defined two separate endpoints `/compare` and `/online_learning`:
1. `/compare` shows the results of our fine-tune models vs. the pretrained model.
2. `/online_learning` takes in `sentence` as inputs and perform [Online learning](https://en.wikipedia.org/wiki/Online_machine_learning)

<b>NOTE:</b> currently `/online_learning` is WIP. Stay tuned!

Start a service with reload enabled:
```python
bentoml serve service:svc --reload
```
With the `--reload` flag, the API server will automatically restart when the source file `service.py` is being updated.

One can then navigate to `127.0.0.1:3000` and interact with Swagger UI.
One can also verify the endpoints locally with `curl`:
```bash
curl -X POST "http://localhost:3000/compare" \
     -H "accept: application/json" \
     -H "Content-Type: text/plain" \
     -d "\" I love you\""
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
[21:56:07] WARNING  [boot] /Users/aarnphm/mambaforge/lib/python3.9/site-packages/jax/_src/lib/__init__.py:32: UserWarning: JAX on Mac ARM machines is experimental and minimally tested. Please see https://github.com/google/jax/issues/5501 in the event of problems.
                      warnings.warn("JAX on Mac ARM machines is experimental and minimally tested. "

[21:56:10] INFO     [boot] Building BentoML service "online_learning_ft:e5wyr7euksn5fgxi" from build context
                    "/Users/aarnphm/Documents/cs/github/gallery/transformers/online_learning"
           INFO     [boot] Packing model "drobert_ft:y3nplreukohglgxi" from "/Users/aarnphm/bentoml/models/drobert_ft/y3nplreukohglgxi"
           INFO     [boot] Packing model "emotion_distilroberta_base:zdbcndeukohglgxi" from
                    "/Users/aarnphm/bentoml/models/emotion_distilroberta_base/zdbcndeukohglgxi"
           INFO     [boot]
                    ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
                    ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
                    ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
                    ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
                    ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
                    ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

           INFO     [boot] Successfully built Bento(tag="online_learning_ft:e5wyr7euksn5fgxi") at
                    "/Users/aarnphm/bentoml/bentos/online_learning_ft/e5wyr7euksn5fgxi/"
```

This Bento now can be served with `--production`:
```bash
bentoml serve online_learning_ft:latest --production
```

#### Containerize a Bento

Make sure Docker and daemon is running, then `bentoml containerize` will build
a docker image for the model server aboved:
```bash
» bentoml containerize online_learning_ft:latest
# results image tag: online_learning_ft:zt4vvsurw63thgxi
```

Test out the newly built docker image:
```bash
docker run -p 3000:3000 online_learning_ft:zt4vvsurw63thgxi
```

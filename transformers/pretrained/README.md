## Pretrained Roberta with Transformers :handshake: BentoML

<!--
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/pretrained/pretrained_roberta.ipynb)[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/pretrained/pretrained_roberta.ipynb)[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://github.com/bentoml/gallery/tree/main/transformers/pretrained/pretrained_roberta.ipynb)
-->
<div align='center'>
    <p align='center'>
        <a href="https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/pretrained/pretrained_roberta.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
        <a href="https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/pretrained/pretrained_roberta.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="nbviewer"/></a>
        <a href="https://github.com/bentoml/gallery/tree/main/transformers/pretrained/pretrained_roberta.ipynb"><img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter" alt="Made with Jupyter"/></a>
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
async def batch_sentiment(sentence: t.Dict[str, t.List[str]]) -> t.Dict[int, t.Dict[str, t.Union[str, float]]]:
    processed = preprocess(sentence)
    outputs = await asyncio.gather(*[clf_runner.async_run(s) for s in processed["text"]])
    return postprocess(processed, outputs)  # type: ignore


@svc.api(input=JSON(), output=JSON())
async def batch_all_scores(sentence: t.Dict[str, t.List[str]]) -> t.Dict[int, t.Dict[str, t.Union[str, float]]]:
    processed = preprocess(sentence)
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
  lock_packages: false
  packages:
    - -f https://download.pytorch.org/whl/cpu/torch_stable.html
    - torch==1.10.2+cpu
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
» bentoml containerize pretrained_clf:latest
[14:15:48] INFO     [boot] Building docker image for Bento(tag="pretrained_clf:zt4vvsurw63thgxi")...
[+] Building 135.9s (19/19) FINISHED
 => [internal] load build definition from Dockerfile                                                                                                                                                                                                                                   0.0s
 => => transferring dockerfile: 958B                                                                                                                                                                                                                                                   0.0s
 => [internal] load .dockerignore                                                                                                                                                                                                                                                      0.0s
 => => transferring context: 2B                                                                                                                                                                                                                                                        0.0s
 => [internal] load metadata for docker.io/bentoml/bento-server:1.0.0a4-python3.9-debian-runtime                                                                                                                                                                                       1.5s
 => [auth] bentoml/bento-server:pull token for registry-1.docker.io                                                                                                                                                                                                                    0.0s
 => [ 1/13] FROM docker.io/bentoml/bento-server:1.0.0a4-python3.9-debian-runtime@sha256:8db6e54428a0ec4cc5d90710ce2727939ee6c7317255677f4855fc250a597c8c                                                                                                                              14.0s
 => => resolve docker.io/bentoml/bento-server:1.0.0a4-python3.9-debian-runtime@sha256:8db6e54428a0ec4cc5d90710ce2727939ee6c7317255677f4855fc250a597c8c                                                                                                                                 0.0s
 => => sha256:fd58f80ab93165a3528a3517720accedee2888cb041a71dcef39f44ec7950502 5.26kB / 5.26kB                                                                                                                                                                                         0.0s
 => => sha256:72a69066d2febc34d8f3dbcb645f7b851a57e9681322ece7ad8007503b783c19 27.15MB / 27.15MB                                                                                                                                                                                       1.0s
 => => sha256:750772c51f8a5228ee8d1ab13e865af36dac574e65c95d25b985f8cc67340650 92.89MB / 92.89MB                                                                                                                                                                                       4.5s
 => => sha256:6b8f7a6fc9262b707e8de94ecf2a0a79064cef42411a7d06bba5d836e2e0d42a 58.50MB / 58.50MB                                                                                                                                                                                       3.3s
 => => sha256:8db6e54428a0ec4cc5d90710ce2727939ee6c7317255677f4855fc250a597c8c 1.79kB / 1.79kB                                                                                                                                                                                         0.0s
 => => sha256:0aa5f948503b1c0bbdfaca80d9934d064b978a1e008b0fd243af5c568c21cfe1 34.05MB / 34.05MB                                                                                                                                                                                       3.2s
 => => extracting sha256:72a69066d2febc34d8f3dbcb645f7b851a57e9681322ece7ad8007503b783c19                                                                                                                                                                                              2.2s
 => => sha256:5dc30d464099737777f848b3b8e7c5df3d78b9b2da336eb2e391ebb7fb5c0b9d 530B / 530B                                                                                                                                                                                             3.4s
 => => sha256:0c05293a0c9c23ef73fa271d253904ebba7b5100ef2f0fd0a6a33989c39bba35 31.52MB / 31.52MB                                                                                                                                                                                       4.9s
 => => extracting sha256:750772c51f8a5228ee8d1ab13e865af36dac574e65c95d25b985f8cc67340650                                                                                                                                                                                              4.6s
 => => extracting sha256:6b8f7a6fc9262b707e8de94ecf2a0a79064cef42411a7d06bba5d836e2e0d42a                                                                                                                                                                                              2.3s
 => => extracting sha256:0aa5f948503b1c0bbdfaca80d9934d064b978a1e008b0fd243af5c568c21cfe1                                                                                                                                                                                              0.9s
 => => extracting sha256:5dc30d464099737777f848b3b8e7c5df3d78b9b2da336eb2e391ebb7fb5c0b9d                                                                                                                                                                                              0.0s
 => => extracting sha256:0c05293a0c9c23ef73fa271d253904ebba7b5100ef2f0fd0a6a33989c39bba35                                                                                                                                                                                              1.0s
 => [internal] load build context                                                                                                                                                                                                                                                     10.5s
 => => transferring context: 332.01MB                                                                                                                                                                                                                                                 10.4s
 => [ 2/13] RUN groupadd -g 1034 -o bentoml && useradd -m -u 1034 -g 1034 -o -r bentoml                                                                                                                                                                                                0.7s
 => [ 3/13] RUN mkdir /home/bentoml/bento && chown bentoml:bentoml /home/bentoml/bento -R                                                                                                                                                                                              0.3s
 => [ 4/13] WORKDIR /home/bentoml/bento                                                                                                                                                                                                                                                0.0s
 => [ 5/13] COPY --chown=bentoml:bentoml ./env ./env                                                                                                                                                                                                                                   0.0s
 => [ 6/13] RUN chmod +x ./env/docker/init.sh                                                                                                                                                                                                                                          0.2s
 => [ 7/13] RUN ./env/docker/init.sh ensure_python                                                                                                                                                                                                                                     0.5s
 => [ 8/13] RUN ./env/docker/init.sh restore_conda_env                                                                                                                                                                                                                                 0.3s
 => [ 9/13] RUN ./env/docker/init.sh install_pip_packages                                                                                                                                                                                                                            116.6s
 => [10/13] RUN ./env/docker/init.sh install_wheels                                                                                                                                                                                                                                    0.3s
 => [11/13] RUN ./env/docker/init.sh user_setup_script                                                                                                                                                                                                                                 0.3s
 => [12/13] COPY --chown=bentoml:bentoml . ./                                                                                                                                                                                                                                          0.3s
 => [13/13] RUN chmod +x ./env/docker/entrypoint.sh                                                                                                                                                                                                                                    0.2s
 => exporting to image                                                                                                                                                                                                                                                                 0.7s
 => => exporting layers                                                                                                                                                                                                                                                                0.6s
 => => writing image sha256:8423510b91a7d646145b6470a83c182bf1ffe635566155d43f0d8da01fc9ca8a                                                                                                                                                                                           0.0s
 => => naming to docker.io/library/pretrained_clf:zt4vvsurw63thgxi                                                                                                                                                                                                                     0.0s
[14:18:04] INFO     [boot] Successfully built docker image "pretrained_clf:zt4vvsurw63thgxi"
```

Test out the newly built docker image:
```bash
docker run -p 3000:3000 pretrained_clf:zt4vvsurw63thgxi
```

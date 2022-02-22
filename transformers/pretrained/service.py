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

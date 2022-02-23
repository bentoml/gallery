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

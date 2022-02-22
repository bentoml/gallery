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

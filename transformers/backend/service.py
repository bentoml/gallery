import typing as t

import bentoml

from config import TASKS, MODEL_NAME, TRANSFER_LEARNING_TAG, RUNNER_NAME

from bentoml.io import Text, JSON


classifier_all_scores = bentoml.transformers.load_runner(f'{MODEL_NAME}:mkehhfen6cfw5gxi', name=RUNNER_NAME, tasks=TASKS, return_all_scores=True)
transfer_learning_runner = bentoml.transformers.load_runner(TRANSFER_LEARNING_TAG, name="roberta_transfer_learning", tasks=TASKS, return_all_scores=True)

svc = bentoml.Service(
    name="transformers_roberta_text_classification",
    runners=[classifier_all_scores, transfer_learning_runner],
)


@svc.api(input=Text(), output=JSON())
async def sentiment(sentence: str) -> t.List[t.Dict[str, t.Any]]:
    output = await classifier_all_scores.async_run(sentence)
    return output

@svc.api(input=Text(), output=JSON())
def transfer_learning(sentences: t.List[str]) -> t.List[t.Dict[str, t.Any]]:
    output = transfer_learning_runner.run_batch(sentences)
    return output


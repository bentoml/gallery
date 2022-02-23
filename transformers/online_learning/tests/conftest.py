# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import typing as t

import pytest
from bentoml.testing.server import run_api_server


def pytest_configure(config):  # pylint: disable=unused-argument
    import bentoml
    import transformers

    FT_NAME = "drobert_ft"
    PRETRAINED = "emotion_distilroberta_base"

    # transformers model #
    FINETUNE_MODEL = "aarnphm/finetune_emotion_distilroberta"
    PRETRAINED_MODEL = "j-hartmann/emotion-english-distilroberta-base"

    m1 = transformers.AutoModelForSequenceClassification.from_pretrained(FINETUNE_MODEL)
    t1 = transformers.AutoTokenizer.from_pretrained(FINETUNE_MODEL)
    _ = bentoml.transformers.save(FT_NAME, m1, tokenizer=t1)

    m2 = transformers.AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL
    )
    t2 = transformers.AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    _ = bentoml.transformers.save(PRETRAINED, m2, tokenizer=t2)


@pytest.fixture(scope="session")
def host() -> t.Generator[str, None, None]:
    import sys
    import subprocess

    cmd = f"{sys.executable} -m bentoml build"
    subprocess.run(cmd, shell=True, check=True)

    with run_api_server(bento="online_learning_ft:latest") as host:
        yield host

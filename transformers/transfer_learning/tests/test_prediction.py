# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import logging
import typing as t
import random

import pytest
from bentoml.testing.utils import async_request

logger = logging.getLogger("bentoml")

random.seed(400)

BATCH = 5

@pytest.fixture()
def batched_sentence() -> t.Callable[[int], str]:
    def _(batch=5) -> str:
        from essential_generators import DocumentGenerator
        gen = DocumentGenerator()
        return str([gen.sentence() for _ in range(batch)])
    return _


@pytest.mark.asyncio
async def test_sentiment(host, batched_sentence):
    inputs = batched_sentence(BATCH)
    logger.info(inputs)
    await async_request(
        "POST",
        f"http://{host}/sentiment",
        headers={"Content-Type": "application/json"},
        data=inputs,
        assert_status=200,
    )


@pytest.mark.asyncio
async def test_all_scores(host, batched_sentence):
    inputs = batched_sentence(BATCH)
    await async_request(
        "POST",
        f"http://{host}/all_scores",
        headers={"Content-Type": "application/json"},
        data=inputs,
        assert_status=200,
    )

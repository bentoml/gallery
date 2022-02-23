# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import json
import random
import typing as t
import logging

import pytest
from bentoml.testing.utils import async_request

logger = logging.getLogger("bentoml")

random.seed(400)

BATCH = 5

expected_results = b'{"input":"\\"Our path diverges.\\"","label":"neutral"}'


@pytest.fixture()
def batched_sentence() -> t.Callable[[int], str]:
    def _(batch=5) -> str:
        from essential_generators import DocumentGenerator

        gen = DocumentGenerator()
        return json.dumps({"text": [gen.sentence() for _ in range(batch)]})

    return _


@pytest.mark.asyncio
async def test_sentiment(host):
    await async_request(
        "POST",
        f"http://{host}/sentiment",
        headers={"Content-Type": "application/json"},
        data=json.dumps("Our path diverges."),
        assert_status=200,
        assert_data=expected_results,
    )


@pytest.mark.asyncio
async def test_batch_sentiment(host, batched_sentence):
    inputs = batched_sentence(BATCH)
    await async_request(
        "POST",
        f"http://{host}/batch_sentiment",
        headers={"Content-Type": "application/json"},
        data=inputs,
        assert_status=200,
    )


@pytest.mark.asyncio
async def test_batch_all_scores(host, batched_sentence):
    inputs = batched_sentence(BATCH)
    await async_request(
        "POST",
        f"http://{host}/batch_all_scores",
        headers={"Content-Type": "application/json"},
        data=inputs,
        assert_status=200,
    )

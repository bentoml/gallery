# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import json
import random
import typing as t

import pytest
from bentoml.testing.utils import async_request

random.seed(400)


@pytest.mark.asyncio
async def test_compare(host: t.Generator[str, None, None]) -> None:
    inputs = json.dumps("I love you")
    await async_request(
        "POST",
        f"http://{host}/compare",
        headers={"Content-Type": "application/json"},
        data=inputs,
        assert_status=200,
    )

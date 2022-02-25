# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import subprocess
import typing as t

import pytest
from bentoml.testing.server import run_api_server


def pytest_configure(config):  # pylint: disable=unused-argument
    import os
    import sys

    cmd = f"{sys.executable} {os.path.join(os.getcwd(), 'train.py')}"
    subprocess.run(cmd, shell=True, check=True)


@pytest.fixture(scope="session")
def host() -> t.Generator[str, None, None]:
    import bentoml

    subprocess.run("bentoml build", shell=True, check=True)

    with run_api_server(
        bento="tensorflow_mnist_demo:latest",
    ) as host:
        yield host

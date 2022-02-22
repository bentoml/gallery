# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import typing as t

import pytest
from bentoml.testing.server import run_api_server


def pytest_configure(config):  # pylint: disable=unused-argument
    import os
    import sys
    import subprocess

    cmd = f"{sys.executable} {os.path.join(os.getcwd(), 'import_model.py')}"
    subprocess.run(cmd, shell=True, check=True)


@pytest.fixture(scope="session")
def host() -> t.Generator[str, None, None]:
    import bentoml

    bentoml.build("service:svc")

    with run_api_server(bento="pretrained_clf:latest") as host:
        yield host
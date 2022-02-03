import typing as t

import bentoml
import numpy as np
import PIL.Image

from bentoml.io import Image, NumpyNdarray
from PIL.Image import Image as PILImage

{{ cookiecutter.__project_slug }}_runner = bentoml.{{ cookiecutter.__project_dir }}.load_runner(
    "{{ cookiecutter.__full_name }}",
    name="{{ cookiecutter.__project_slug }}_runner",
)

svc = bentoml.Service(
    name="{{ cookiecutter.__full_name }}",
    runners=[
        {{ cookiecutter.__project_slug }}_runner,
    ],
)


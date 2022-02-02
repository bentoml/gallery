import typing as t

import bentoml
import numpy as np
import PIL.Image

from bentoml.io import Image, NumpyNdarray
from PIL.Image import Image as PILImage

roberta_text_classification_runner = bentoml.transformers.load_runner(
    "transformers_roberta_text_classification",
    name="roberta_text_classification_runner",
)

svc = bentoml.Service(
    name="transformers_roberta_text_classification",
    runners=[
        roberta_text_classification_runner,
    ],
)


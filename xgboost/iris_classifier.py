import typing

import bentoml
import xgboost
from bentoml.io import NumpyNdarray, File

if typing.TYPE_CHECKING:
    import numpy as np

agaricus_runner = bentoml.xgboost.load_runner("agaricus:latest")

svc = bentoml.Service("agaricus", runners=[agaricus_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: "np.ndarray") -> "np.ndarray":
    return agaricus_runner.run(input_data)

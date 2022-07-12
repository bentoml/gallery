import bentoml

import mlflow
import torch

mnist_runner = bentoml.mlflow.get('mlflow-pytorch-mnist:latest').to_runner()

svc = bentoml.Service('mlflow_pytorch_mnist', runners=[ mnist_runner ])

input_spec = bentoml.io.NumpyNdarray(
    dtype="float32",
    shape=[-1, 1, 28, 28],
    enforce_shape=True,
    enforce_dtype=True,
)

@svc.api(input=input_spec, output=bentoml.io.NumpyNdarray())
def predict(input_arr):
    return mnist_runner.predict.run(input_arr)

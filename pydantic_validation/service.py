import numpy as np
import pandas as pd
import bentoml
from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel

iris_clf_runner = bentoml.sklearn.get("iris_clf_with_feature_names:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

class IrisFeatures(BaseModel):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float

    # Use custom Pydantic config for additional validation options
    class Config:
        extra = 'forbid'


input_spec = JSON(pydantic_model=IrisFeatures)

@svc.api(input=input_spec, output=NumpyNdarray())
def classify(input_data: IrisFeatures) -> np.ndarray:
    input_df = pd.DataFrame([input_data.dict()])
    return iris_clf_runner.predict.run(input_df)

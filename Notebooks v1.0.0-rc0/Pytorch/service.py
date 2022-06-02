
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray,Text, Image

model_tag = "cancer_classifier_rc0:latest"

# Load the runner for the latest Pytorch model we just saved
cancer_runner = bentoml.pytorch.get(model_tag).to_runner()
data = bentoml.models.get(model_tag)

nn = bentoml.Service("cancer_classifier_rc0", runners=[cancer_runner])

@nn.api(input=NumpyNdarray(), output=Text())
def predict_cancer(input_series: np.ndarray) -> str:    
    try:
        result = cancer_runner.run(input_series)
        result = data.custom_objects['labels'][np.argmax(result.detach().numpy()).item()]
        return result
    except:
        return 'Exception: Invalid Input'

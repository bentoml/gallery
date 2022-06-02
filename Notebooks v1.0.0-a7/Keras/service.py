
# service.py
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Text, Image
import PIL.Image

model_tag = "cifar10_classifier:latest"
# Load the runner for the latest Keras model we just saved
cifar10_runner = bentoml.keras.load_runner(model_tag)
cifar10_model = bentoml.models.get(model_tag)

# Create the cifar10 service with the Keras runner
# Multiple runners may be specified if needed in the runners array
# When packaged as a bento, the runners here will included
cnn = bentoml.Service("cifar10_classifier", runners=[cifar10_runner])

# Create API function with pre- and post- processing logic with your new "cnn" annotation
@cnn.api(input=NumpyNdarray(), output=Text())
def predict_array(input_series: np.ndarray) -> str:   
    try:
        # Define pre-processing logic
        input_data = cifar10_model.custom_objects['preprocessing'](
            input_series)
        
        result = cifar10_runner.run(input_data)
        
        # Define post-processing logic
        result = cifar10_model.custom_objects['labels'][np.argmax(result)]
        return result
    except:
        return 'Exception: Inappropriate input'


@cnn.api(input=Image(), output=Text())
def predict_image(f: PIL.Image) -> "np.ndarray":
    try:
        arr = np.array(f)
        input_data = cifar10_model.custom_objects['preprocessing'](arr)
        result = cifar10_runner.run(input_data)
        # Define post-processing logic
        result = cifar10_model.custom_objects['labels'][np.argmax(result)]
        return result
    except:
        return 'Exception: Invalid input'

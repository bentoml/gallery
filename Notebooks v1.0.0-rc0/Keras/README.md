# BentoML Keras Tutorial

This is a sample project demonstrating basic usage of BentoML with Keras.

In this project, we will train a classifier model using Keras and the Cifar10 dataset, build an prediction service for serving the trained model via an HTTP server, and containerize the model server as a docker image for production deployment.

## Install Dependencies


```python
!pip install -r requirements.txt
```

## Training the model


```python
import keras 
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from keras.layers import GlobalMaxPooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
```


```python
# Loading the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

## Normalizing the data
def preprocessing_fun(data):
    return data/255

x_train_features = preprocessing_fun(x_train)
x_test_features = preprocessing_fun(x_test)

# Defining the neural network
model = keras.Sequential()

# input layer
model.add(Input(shape=x_train[0].shape))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
 
model.add(Flatten())
model.add(Dropout(0.2))
 
# Hidden layer
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
 
# last hidden layer i.e.. output layer
model.add(Dense(10, activation='softmax'))
print(model.summary())

# Compile
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

## Training and fitting the model
model.fit(
      x_train_features, 
      y_train.flatten(), 
      validation_data=(x_test_features, y_test.flatten()), 
      batch_size=32,
      epochs=3)
```

## Save the model instance `model` to BentoML local model store


```python
metadata={'loss': 0.8530 , 'accuracy': 0.7044 , 'val_loss': 0.8454 , 'val_accuracy': 0.7124}

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
custom_obj={'labels': labels,
      'preprocessing':preprocessing_fun}

import bentoml
tag = bentoml.keras.save_model('cifar10_classifier_rc0',
                           model,
                           metadata=metadata,
                           custom_objects = custom_obj)
tag
```

## Create a BentoML Service for serving the model

Note: using `%%writefile` here because bentoml.Service instance must be created in a separate .py file

Here we define as many api endpoints as we want.


```python
%%writefile service.py

# service.py
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Text, Image
import PIL.Image

model_tag = "cifar10_classifier_rc0:latest"
# Load the runner for the latest Keras model we just saved
cifar10_runner = bentoml.keras.get(model_tag).to_runner()
cifar10_model = bentoml.models.get(model_tag)

# Create the cifar10 service with the Keras runner
# Multiple runners may be specified if needed in the runners array
# When packaged as a bento, the runners here will included
cnn = bentoml.Service("cifar10_classifier_rc0", runners=[cifar10_runner])

# Create API function with pre- and post- processing logic with your new "cnn" annotation
@cnn.api(input=NumpyNdarray(), output=Text())
def predict_array(input_series: np.ndarray) -> str:   
    try:
        # Define pre-processing logic
        input_data = cifar10_model.custom_objects['preprocessing'](
            input_series)
        
        result = cifar10_runner.predict.run(input_data)
        
        # Define post-processing logic
        result = cifar10_model.custom_objects['labels'][np.argmax(result)]
        return result
    except:
        return 'Exception: Inappropriate input'


@cnn.api(input=Image(), output=Text())
def predict_image(f: PIL.Image) -> "np.ndarray":
    try:
        arr = np.expand_dims(np.array(f),0)
        input_data = cifar10_model.custom_objects['preprocessing'](arr)
        result = cifar10_runner.predict.run(input_data)
        # Define post-processing logic
        result = cifar10_model.custom_objects['labels'][np.argmax(result)]
        return result
    except:
        return 'Exception: Invalid input'
```

Start a dev model server to test out the service defined above


```python
!bentoml serve service.py:cnn 
```

Open your web browser at http://127.0.0.1:3000 to view the Bento UI for sending test requests. Now you can use something like:

curl -H "Content-Type: multipart/form-data" -F'fileobj=@sample_image.png;type=image/png' http://127.0.0.1:3000/predict_image

or execute the following code snippet.


```python
## Array data

import requests,json 
def test_numpy(host, img_data):
    img_json=json.dumps(img_data.tolist())
    print('Sending Request')
    resp = requests.post(
        url = f"http://{host}/predict_array",
        headers={"Content-Type": "application/json"},
        data=img_json,
           )
    print('Response')
    return resp


## Image Data
from PIL import Image 

def test_image(host, img_path):
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    print('Sending Request')
    resp = requests.post(
        url = f"http://{host}/predict_image",
        headers={"Content-Type": "image/png"},
        data=img_bytes,
           )
    print('Response')
    return resp


```


```python
arr = np.expand_dims(x_test[2100],0)
response=test_numpy('127.0.0.1:3000', arr)
print(response.text)

img_path = f"sample_image.png"
response=test_image('127.0.0.1:3000',img_path)
response.text

```

## Build a Bento for distribution and deployment

Bento is the distribution format in BentoML which captures all the source code, model files, config files and dependency specifications required for running the service for production deployment. Think of it as Docker/Container designed for machine learning models.

Create a bento file `bentofile.yaml` for building a Bento for the service:



```python
%%writefile bentofile.yaml

service: "service.py:cnn"  # A convention for locating your service: <YOUR_SERVICE_PY>:<YOUR_SERVICE_ANNOTATION>
description: "file: ./README.md"
labels:
    owner: bentoml-team
    stage: demo
include:
 - "*.py"  # A pattern for matching which files to include in the bento
python:
  packages:
   - keras # Additional libraries to be included in the bento
   - numpy
   - Pillow
   - tensorflow
```

Simply run `bentoml build` from current directory to build a Bento with the latest version of the tensorflow_mnist model. This may take a while when running for the first time for BentoML to resolve all dependency versions:


```python
!bentoml build
```

Starting a dev server with the Bento build:


```python
!bentoml serve cifar10_classifier_rc0:latest
```

## Containerize and Deployment

Bento is designed to be deployed to run efficiently in a variety of different environments. And there are lots of deployment options and tools as part of the BentoML eco-system, such as Yatai and bentoctl for direct deployment to cloud platforms.

In this guide, we will show you the most basic way of deploying a Bento, which is converting a Bento into a Docker image containing the HTTP model server.

Make sure you have docker installed and docker deamon running, and run the following command:


```python
!bentoml containerize cifar10_classifier_rc0:latest
```

This will build a new docker image with all source code, model files and dependencies in place, and ready for production deployment. To start a container with this docker image locally, run:

`docker run -p 3000:3000 cifar10_classifier_rc0:c5nnhiw7666ijgh2 `

## What's Next?,
   
  - 👉 [Pop into our Slack community!](https://l.linklyhq.com/l/ktO8) We're happy to help with any issue you face or even just to meet you and hear what you're working on.,
   
  - Dive deeper into the [Core Concepts](https://docs.bentoml.org/en/latest/concepts/index.html) in BentoML,
  
  - Learn how to use BentoML with other ML Frameworks at [Frameworks Guide](https://docs.bentoml.org/en/latest/frameworks/index.html) or check out other [gallery projects](https://github.com/bentoml/gallery),
  - Learn more about model deployment options for Bento:,
      - [🦄️ Yatai](https://github.com/bentoml/Yatai): Model Deployment at scale on Kubernetes,
      - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on any cloud platform

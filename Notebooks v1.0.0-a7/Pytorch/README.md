# BentoML Pytorch Tutorial

This is a sample project demonstrating basic usage of BentoML with Pytorch.

In this project, we will train a classifier model using Pytorch and the breast cancer dataset, build an prediction service for serving the trained model via an HTTP server, and containerize the model server as a docker image for production deployment.

## Install Dependencies


```python
!pip install -r requirements.txt
```

## Training the model


```python
# Loading the dataset
from sklearn import datasets
dataset = datasets.load_breast_cancer()

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    test_size=0.2)

# Converting to Pytorch Tensors
Xtrain = torch.from_numpy(X_train).float()
Xtest = torch.from_numpy(X_test).float()
Ytrain = torch.from_numpy(Y_train)
Ytest = torch.from_numpy(Y_test)

# Defining the neural network
import torch.nn as nn
import torch.nn.functional as F
input_size = Xtrain.shape[1]
output_size = len(Ytrain.unique())

class Net(nn.Module):
    
    def __init__(self): 
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 100) 
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_size)        
        self.hidden_size = 100
        self.activation_fn = 'relu'               
    
    def forward(self, x):
        
        activation_fn = F.relu

        x = activation_fn(self.fc1(x))
        x = activation_fn(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim = -1)
    
# Training and evaluating the model
import torch.optim as optim

def train_and_evaluate_model(model, learn_rate=0.001):
    epoch_data = []
    epochs = 1001
    
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)    
    loss_fn = nn.NLLLoss()    
    test_accuracy = 0.0
    
    for epoch in range(1, epochs):
        optimizer.zero_grad()
        Ypred = model(Xtrain)
        loss = loss_fn(Ypred , Ytrain)
        loss.backward()
        optimizer.step()

        Ypred_test = model(Xtest)
        loss_test = loss_fn(Ypred_test, Ytest)
        _, pred = Ypred_test.data.max(1)

        test_accuracy = pred.eq(Ytest.data).sum().item() / Y_test.size        
        epoch_data.append([epoch, loss.data.item(), loss_test.data.item(), test_accuracy])

        if epoch % 100 == 0:
            print ('epoch - %d train loss - %.2f test loss - %.2f Test accuracy - %.4f'\
                   % (epoch, loss.data.item(), loss_test.data.item(), test_accuracy))
            

    return {'model' : model,
            'epoch_data' : epoch_data, 
            'num_epochs' : epochs, 
            'optimizer' : optimizer, 
            'loss_fn' : loss_fn,
            'test_accuracy' : test_accuracy,
            '_, pred' : Ypred_test.data.max(1),
            'actual_test_label' : Ytest,
            }

net = Net()
result = train_and_evaluate_model(net)
```


```python
from sklearn.metrics import accuracy_score, recall_score, precision_score

accuracy=accuracy_score(Ytest,result['_, pred'][1])
recall=recall_score(Ytest,result['_, pred'][1])
precision=precision_score(Ytest,result['_, pred'][1])
```

## Save the model instance `net` to BentoML local model store


```python
metadata ={'Accuracy':accuracy,'Precision':precision,'Recall':recall}

custom_objects = {'labels':['Malignant','Benign']}

import bentoml
tag = bentoml.pytorch.save('cancer_classifier',
                           net,
                           metadata=metadata,
                           custom_objects = custom_objects)
tag
```

## Create a BentoML Service for serving the model

Note: using `%%writefile` here because bentoml.Service instance must be created in a separate .py file

Here we define as many api endpoints as we want.


```python
%%writefile service.py

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray,Text, Image

model_tag = "cancer_classifier:latest"

# Load the runner for the latest Pytorch model we just saved
cancer_runner = bentoml.pytorch.load_runner(model_tag)
data = bentoml.models.get(model_tag)

nn = bentoml.Service("cancer_classifier", runners=[cancer_runner])

@nn.api(input=NumpyNdarray(), output=Text())
def predict_cancer(input_series: np.ndarray) -> str:    
    try:
        result = cancer_runner.run(input_series)
        result = data.custom_objects['labels'][np.argmax(result).item()]
        return result
    except:
        return 'Exception: Invalid Input'
```

Start a dev model server to test out the service defined above


```python
!bentoml serve service.py:svc --reload
```

Open your web browser at http://127.0.0.1:3000 to view the Bento UI for sending test requests. Now you can use something like:


```python
import requests,json 

def predict_cancer(host, data):
    data_json=json.dumps(data.tolist())
    print('Sending Request')
    resp = requests.post(
        url = f"http://{host}/predict_cancer",
        headers={"Content-Type": "application/json"},
        data=data_json,
           )
    print('Response')
    return resp

```


```python
response = predict_cancer('127.0.0.1:3000', X_test[21])
response.text
```

## Build a Bento for distribution and deployment

Bento is the distribution format in BentoML which captures all the source code, model files, config files and dependency specifications required for running the service for production deployment. Think of it as Docker/Container designed for machine learning models.

Create a bento file `bentofile.yaml` for building a Bento for the service:



```python
%%writefile bentofile.yaml

service: "service.py:nn"  # A convention for locating your service: <YOUR_SERVICE_PY>:<YOUR_SERVICE_ANNOTATION>
description: "file: ./README.md"
labels:
    owner: bentoml-team
    stage: demo
include:
 - "*.py"  # A pattern for matching which files to include in the bento
python:
  packages:
   - numpy  # Additional libraries to be included in the bento
   - torch
```

Simply run `bentoml build` from current directory to build a Bento with the latest version of the tensorflow_mnist model. This may take a while when running for the first time for BentoML to resolve all dependency versions:


```python
!bentoml build
```

Starting a dev server with the Bento build:


```python
!bentoml serve cancer_classifier:latest
```

## Containerize and Deployment

Bento is designed to be deployed to run efficiently in a variety of different environments. And there are lots of deployment options and tools as part of the BentoML eco-system, such as Yatai and bentoctl for direct deployment to cloud platforms.

In this guide, we will show you the most basic way of deploying a Bento, which is converting a Bento into a Docker image containing the HTTP model server.

Make sure you have docker installed and docker deamon running, and run the following command:


```python
!bentoml containerize cancer_classifier:latest
```

This will build a new docker image with all source code, model files and dependencies in place, and ready for production deployment. To start a container with this docker image locally, run:

`docker run -p 3000:3000 cancer_classifier:rickmtw752h5xgh2 `

## What's Next?,
   
  - 👉 [Pop into our Slack community!](https://l.linklyhq.com/l/ktO8) We're happy to help with any issue you face or even just to meet you and hear what you're working on.,
   
  - Dive deeper into the [Core Concepts](https://docs.bentoml.org/en/v1.0.0-a7/concepts/index.html) in BentoML,
  
  - Learn how to use BentoML with other ML Frameworks at [Frameworks Guide](https://docs.bentoml.org/en/v1.0.0-a7/frameworks/index.html) or check out other [gallery projects](https://github.com/bentoml/gallery),
  - Learn more about model deployment options for Bento:,
      - [🦄️ Yatai](https://github.com/bentoml/Yatai): Model Deployment at scale on Kubernetes,
      - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on any cloud platform

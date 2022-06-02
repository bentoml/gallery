# BentoML Transformer Tutorial

This is a sample project demonstrating basic usage of BentoML with Transformer.

In this project, we will train a classifier model using Transformer and the translation task, build an prediction service for serving the trained model via an HTTP server, and containerize the model server as a docker image for production deployment.

## Install Dependencies


```python
!pip install -r requirements.txt
```

## Training the model


```python
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

model_checkpoint = 't5-small'
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

```

## Save the model instance `model` to BentoML local model store


```python
import bentoml

tag = bentoml.transformers.save('translation', 
                                model, 
                                tokenizer = tokenizer, 
                                metadata = {'Description':'Using AutoModel'}
                              )
tag
```

## Create a BentoML Service for serving the model

Note: using `%%writefile` here because bentoml.Service instance must be created in a separate .py file

Here we define as many api endpoints as we want.


```python
%%writefile service.py

import bentoml
from bentoml.io import Text

model_tag = "translation:latest"

translation_runner = bentoml.transformers.load_runner(model_tag,tasks='translation')

translate = bentoml.Service("translation", runners=[translation_runner])

@translate.api(input=Text(), output=Text())
def translate_text(input_series: str) -> str:
    try:
        result = translation_runner.run(input_series)
        print(result)
        return result['translation_text']
    except:
        return 'Invalid Input'


```

Start a dev model server to test out the service defined above


```python
!bentoml serve service.py:svc --reload
```

Open your web browser at http://127.0.0.1:3000 to view the Bento UI for sending test requests. Now you can use something like:


```python
import requests,json 
def test_translation(host, data):
    data_json=json.dumps(data)
    print('Sending Request')
    resp = requests.post(
        url = f"http://{host}/translate_text",
        headers={"Content-Type": "application/json"},
        data=data_json,
           )

    print('Response')
    return resp
```


```python
response=test_translation('127.0.0.1:3000', 'This too shall pass.')
response.text

```

## Build a Bento for distribution and deployment

Bento is the distribution format in BentoML which captures all the source code, model files, config files and dependency specifications required for running the service for production deployment. Think of it as Docker/Container designed for machine learning models.

Create a bento file `bentofile.yaml` for building a Bento for the service:



```python
%%writefile bentofile.yaml

service: "service.py:translate"  # A convention for locating your service: <YOUR_SERVICE_PY>:<YOUR_SERVICE_ANNOTATION>
description: "file: ./README.md"
labels:
    owner: bentoml-team
    stage: demo
include:
 - "*.py"  # A pattern for matching which files to include in the bento
python:
  packages:
   - transformers[tf-cpu] # Additional libraries to be included in the bento


```

Simply run `bentoml build` from current directory to build a Bento with the latest version of the tensorflow_mnist model. This may take a while when running for the first time for BentoML to resolve all dependency versions:


```python
!bentoml build
```

Starting a dev server with the Bento build:


```python
!bentoml serve translation:latest
```

## Containerize and Deployment

Bento is designed to be deployed to run efficiently in a variety of different environments. And there are lots of deployment options and tools as part of the BentoML eco-system, such as Yatai and bentoctl for direct deployment to cloud platforms.

In this guide, we will show you the most basic way of deploying a Bento, which is converting a Bento into a Docker image containing the HTTP model server.

Make sure you have docker installed and docker deamon running, and run the following command:


```python
!bentoml containerize translation:latest
```

This will build a new docker image with all source code, model files and dependencies in place, and ready for production deployment. To start a container with this docker image locally, run:

`docker run -p 3000:3000 translation:hmto4mhaxk7emcdr `

## What's Next?,
   
  - 👉 [Pop into our Slack community!](https://l.linklyhq.com/l/ktO8) We're happy to help with any issue you face or even just to meet you and hear what you're working on.,
   
  - Dive deeper into the [Core Concepts](https://docs.bentoml.org/en/v1.0.0-a7/concepts/index.html) in BentoML,
  
  - Learn how to use BentoML with other ML Frameworks at [Frameworks Guide](https://docs.bentoml.org/en/v1.0.0-a7/frameworks/index.html) or check out other [gallery projects](https://github.com/bentoml/gallery),
  - Learn more about model deployment options for Bento:,
      - [🦄️ Yatai](https://github.com/bentoml/Yatai): Model Deployment at scale on Kubernetes,
      - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on any cloud platform

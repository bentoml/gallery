# BentoML Transformer Tutorial

This is a sample project demonstrating basic usage of BentoML with Transformer.

In this project, we will train a classifier model using Transformer and the Summarization Pipeline, build an prediction service for serving the trained model via an HTTP server, and containerize the model server as a docker image for production deployment.

## Install Dependencies


```python
!pip install -r requirements.txt
```

## Create the transformer pipeline


```python
from transformers import pipeline

summarizer = pipeline("summarization")
```

## Save the pipeline instance `summarizer` to BentoML local model store


```python
import bentoml

tag = bentoml.transformers.save('summarization', 
                                summarizer,
                                metadata={'Description':'Created using Transformer Pipeline'})
tag
```

## Create a BentoML Service for serving the model

Note: using `%%writefile` here because bentoml.Service instance must be created in a separate .py file

Here we define as many api endpoints as we want.


```python
%%writefile service.py

import bentoml
from bentoml.io import Text

model_tag = "summarization:latest"

summarize_runner = bentoml.transformers.load_runner(model_tag,tasks='summarization')
summarize_model = bentoml.models.get(model_tag)

summarize = bentoml.Service("summarization", runners=[summarize_runner])


@summarize.api(input=Text(), output=Text())
def summarize_text(input_series: str) -> str:
    try:
        result = summarize_runner.run(input_series)
        return result['summary_text']
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
def test_summarization(host, data):
    data_json=json.dumps(data)
    print('Sending Request')
    resp = requests.post(
        url = f"http://{host}/summarize_text",
        headers={"Content-Type": "application/json"},
        data=data_json,
           )

    print('Response')
    return resp
```


```python
ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
response = test_summarization('127.0.0.1:3000',ARTICLE)
response.text

```

## Build a Bento for distribution and deployment

Bento is the distribution format in BentoML which captures all the source code, model files, config files and dependency specifications required for running the service for production deployment. Think of it as Docker/Container designed for machine learning models.

Create a bento file `bentofile.yaml` for building a Bento for the service:



```python
%%writefile bentofile.yaml

service: "service.py:summarize"  # A convention for locating your service: <YOUR_SERVICE_PY>:<YOUR_SERVICE_ANNOTATION>
description: "file: ./README.md"
labels:
    owner: bentoml-team
    stage: demo
include:
 - "*.py"  # A pattern for matching which files to include in the bento
python:
  packages:
   - transformers[tf-cpu]  # Additional libraries to be included in the bento

```

Simply run `bentoml build` from current directory to build a Bento with the latest version of the tensorflow_mnist model. This may take a while when running for the first time for BentoML to resolve all dependency versions:


```python
!bentoml build
```

Starting a dev server with the Bento build:


```python
!bentoml serve summarization:latest
```

## Containerize and Deployment

Bento is designed to be deployed to run efficiently in a variety of different environments. And there are lots of deployment options and tools as part of the BentoML eco-system, such as Yatai and bentoctl for direct deployment to cloud platforms.

In this guide, we will show you the most basic way of deploying a Bento, which is converting a Bento into a Docker image containing the HTTP model server.

Make sure you have docker installed and docker deamon running, and run the following command:


```python
!bentoml containerize summarization:latest
```

This will build a new docker image with all source code, model files and dependencies in place, and ready for production deployment. To start a container with this docker image locally, run:

`docker run -p 3000:3000 cifar10_classifier:g3fbsno5u6agfgh2 `

## What's Next?,
   
  - 👉 [Pop into our Slack community!](https://l.linklyhq.com/l/ktO8) We're happy to help with any issue you face or even just to meet you and hear what you're working on.,
   
  - Dive deeper into the [Core Concepts](https://docs.bentoml.org/en/latest/concepts/index.html) in BentoML,
  
  - Learn how to use BentoML with other ML Frameworks at [Frameworks Guide](https://docs.bentoml.org/en/latest/frameworks/index.html) or check out other [gallery projects](https://github.com/bentoml/gallery),
  - Learn more about model deployment options for Bento:,
      - [🦄️ Yatai](https://github.com/bentoml/Yatai): Model Deployment at scale on Kubernetes,
      - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on any cloud platform

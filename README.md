[<img src="https://raw.githubusercontent.com/bentoml/BentoML/main/docs/source/_static/img/bentoml-readme-header.jpeg" width="600px" margin-left="-5px">](https://github.com/bentoml/BentoML)

# BentoML Gallery  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=BentoML:%20The%20Unified%20Model%20Serving%20Framework%20&url=https://github.com/bentoml&via=bentomlai&hashtags=mlops,bentoml)

BentoML is an open platform that simplifies ML model deployment and enables you to serve your models at production scale in minutes

👉 [Pop into our Slack community!](https://join.slack.bentoml.org) We're happy to help with any issue you face or even just to meet you and hear what you're working on :)

This repository is a collection of machine learning projects that utilizes [BentoML](https://github.com/bentoml/BentoML)
for model serving. The goal is to demonstrate real-world BentoML usage and best practices in
productionizing a machine learning model for serving.

Note: You are looking at gallery examples for BentoML 1.0 version, which is still under early beta release. 
For prior stable versions (0.13.x), see the [0.13-LTS branch](https://github.com/bentoml/gallery/tree/0.13-LTS).


BentoML 1.0 preview release is required for running gallery projects here:

```bash
pip install bentoml --pre
```


## Projects List

* Scikit-learn Iris Classifier: https://github.com/bentoml/gallery/tree/main/quickstart
* PyTorch MNIST: https://github.com/bentoml/gallery/tree/main/pytorch_mnist


## Project layout

Each gallery project is under its own folder, typically containing the following files:

| file name | description |
| --- | --- |
| README.md | a step-by-step guide running the project Python scripts from CLI |
| {PROJECT_NAME}.ipynb | a jupyter notebook shows the same workflow but from notebook environment |
| requirements.txt | required PyPI packages for this project |
| train.py | a python script for training an ML model and saving it with BentoML |
| import_model.py | import an existing trained model to BentoML |
| service.py | python code that defines the bentoml.Service instance for serving |
| bentofile.yaml | the bento build file for building the service into a Bento |
| .bentoignore | files to exclude from build directory, when building a Bento |
| benchmark.py | a python script that tests the baseline performance of the final model server created |


## How to contribute

If you have issues running these projects or have suggestions for improvement, use [Github Issues 🐱](https://github.com/bentoml/gallery/issues/new)

If you are interested in contributing new projects to this repo, let's talk 🥰 - Join us on [Slack](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg) and share your idea in #bentoml-dev channel

Before you create a Pull Request, make sure:
* Follow the basic structures and naming conventions of other existing gallery projects
* Ensure your project runs with the latest version of BentoML


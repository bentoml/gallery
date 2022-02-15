# BentoML's Sentiment Analysis Tutorial

This is a sample project demonstrating basic usage of BentoML, The Unified Model
Serving Framework.

In this project, There are two parts:

1. We will import a pretrained Roberta model
2. We will also fine-tuning the model and compare the results between the two models.

We then create an ML service for both models, serve it behind an HTTP endpoint, and containerize the model
server as a docker image for production deployment.

We will also enable GPU supports for this service, and deploy it to [Yatai](https://github.com/bentoml/Yatai)

### Specification

model: [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english)
framework: [`transformers`](https://huggingface.co/docs/transformers/index) + [`bentoml`](https://github.com/bentoml/BentoML)
transfer learning: [Source](https://github.com/bentoml/gallery/tree/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb) | [nbviewer](https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb) | [Colab](https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb)

### Install Dependencies

Install python packages required for running this project:
```bash
pip install -r ./requirements-dev.txt
```


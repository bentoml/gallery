<h1 align='center'>
    <img src="./_static/bentoml.svg" style="height: 50px; max-width: 20%;"> <img src="./_static/handshake.svg" style="height: 50px; max-width: 20%;"> <img src="./_static/huggingface_logo.svg" style="height: 50px; max-width: 20%;">
</h1>

This project demonstrates basic usage of BentoML, The Unified Model
Serving Framework with Transformers.

In this project, There are two parts:

1. We will import a pretrained Roberta model
2. We will also fine-tuning the model and compare the results between the two models.

We then create an ML service for both models, serve it behind an HTTP endpoint, and containerize the model
server as a docker image for production deployment.

We will also enable GPU supports for this service, and deploy it to [Yatai](https://github.com/bentoml/Yatai)

### Specification

- model [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english)
- stack [transformers](https://huggingface.co/docs/transformers/index) + [bentoml](https://github.com/bentoml/BentoML)

- fine-tune <a href="https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="nbviewer"/></a> <a href="https://github.com/bentoml/gallery/tree/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb"><img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter" alt="Made with Jupyter"/></a>

### Install Dependencies

Install python packages required for running this project:
```bash
pip install -r ./requirements-dev.txt
```


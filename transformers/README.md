<h1 align='center'>
    <img src="./_static/bentoml.svg" style="height: 50px; max-width: 20%;"> <img src="./_static/handshake.svg" style="height: 50px; max-width: 20%;"> <img src="./_static/huggingface_logo.svg" style="height: 50px; max-width: 20%;">
</h1>

This folder demonstrates basic usage of BentoML, The Unified Model Serving Framework with Transformers.

There are currently the following projects:

1. [pretrained](./pretrained): How to serve a pretrained Transformers model with BentoML.
2. [gpu_serving](./gpu_serving): How to perform online learning with a fine-tune Transformers model with BentoML (GPU supports).
<!-- TODO: add online learning on new data -->

We will demonstrate how to create an ML service for both models, serve it behind an HTTP endpoint, and containerize the model server as a docker image for production deployment.

Install python packages required under any folder of choice:

```bash
pip install -r ./requirements.txt
```


# BentoML Quickstart Demo

This is a sample project demonstrating basic usage of BentoML, the machine learning model serving library.


1. Train a classification model with sklearn's built-in iris dataset:

```bash
python train.py
```

Verify that the models has saved to local model store:

```bash
bentoml models list
```

Verify that the model can be loaded as runner in Python:

```python
import bentoml.sklearn

runner = bentoml.sklearn.load_runner("iris_clf:latest")

runner.run([5.9, 3. , 5.1, 1.8])  # => array(2)
```

2. Test the service code defined in `iris_classifier.py`:

Start an API server locally:

```bash
bentoml serve iris_classifier:svc --reload
```

With the `--reload` flag, the API server will automatically restart when the source
file `iris_classifier.py` is being edited, to boost your development productivity.


Verify the endpoint can be accessed from localhost:
```bash
curl -X POST -H "content-type: application/json" --data "[5, 4, 3, 2]" http://127.0.0.1:5000/classify
```

3. Build Bento for deployment

A `bentofile` is already created in this directory for building a Bento for the iris_classifier
service:

```yaml
service: "iris_classifier:svc"
description: "file: ./README.md"
labels:
  owner: bentoml-team
  stage: demo
include:
- "*.py"
python:
  packages:
    - scikit-learn
    - pandas
```

Simply run `bentoml build` from current directory to build a Bento with the latest
version of the `iris_clf` model. This may take a while when running for the first
time for BentoML to resolve all dependency versions:

```
> bentoml build

[12/06/2021 17:09:01] INFO     Building BentoML service "iris_classifier:invwzzsw7li6zckb2ie5eubhd" from build context
                               "/Users/chef/workspace/gallery/quickstart"
                      INFO     Packing required models: "iris_clf:7kr6hkcw6ti6zeox2ie5eubhd"
                      INFO     Locking PyPI package versions..
[12/06/2021 17:10:55] INFO     Bento build success, Bento(tag="iris_classifier:invwzzsw7li6zckb2ie5eubhd",
                               path="/Users/chef/bentoml/bentos/iris_classifier/invwzzsw7li6zckb2ie5eubhd/") created
```


4. Containerize Bento for deployment

Make sure you have docker installed and docker deamon running, and run the following command:

```bash
bentoml containerize iris_classifier:latest
```




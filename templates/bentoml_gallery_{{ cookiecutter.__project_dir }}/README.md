# BentoML's {{cookiecutter.framework}} {{cookiecutter.project_name}} Tutorial

This is a sample project demonstrating basic usage of BentoML, The Unified Model
Serving Framework.

In this project, we will train a {{cookiecutter.project_name}} using {{cookiecutter.framework}}, build
an ML service for the model, serve the model behind an HTTP endpoint, and containerize the model
server as a docker image for production deployment.

This project is also available to run from a [Jupyter Notebook](https://github.com/bentoml/gallery/blob/main/{{ cookiecutter.__project_dir }}/{{ cookiecutter.__project_slug }}/{{ cookiecutter.__full_name }}_demo.ipynb). You can also try it out on [Colab](https://colab.research.google.com/github/bentoml/gallery/blob/main/{{ cookiecutter.__project_dir }}/{{ cookiecutter.__project_slug }}/{{ cookiecutter.__full_name }}_demo.ipynb).

### Install Dependencies

Install python packages required for running this project:
```bash
pip install -r ./requirements.txt
```

### Model Training

First step, train the {{cookiecutter.project_name}} with {{cookiecutter.framework}} and then save the model to BentoML local model store:

```bash
python train.py
```

One can check the newly saved model in BentoML local model store:

```bash
bentoml models list
```

Verify that the model can be loaded as runner from an interactive Python shell:

```python
import bentoml
# import necessary library for preprocess

runner = bentoml.{{ cookiecutter.__project_dir }}.load_runner("{{ cookiecutter.__project_slug }}:latest")

# preprocess an input called `inp`
...

runner.run(inp)
```

### Create ML Service

The ML Service code is defined in the [`service.py`](./service.py) file:

```python
{% include 'service.py' %}
```

We defined the following API for our endpoints with a single runner: [[endpoints]]

Start an API server locally to test the service code above:

```bash
bentoml serve service:svc --reload
```

With the `--reload` flag, the API server will automatically restart when the source
file `service.py` is being edited, to boost your development productivity.


Verify the endpoint can be accessed locally via `curl`:
```bash
curl -H "Content-Type: multipart/form-data" -F'fileobj=@samples/1.png;type=image/png' http://127.0.0.1:5000/[[an_endpoint]]
```


### Build Bento for deployment

A [`bentofile`](./bentofile.yaml) is already created in this directory for building a
Bento for the service:

```yaml
{% include 'bentofile.yaml' %}
```

Note that we exclude `tests/` from the bento using `exclude`.

Simply run `bentoml build` from current directory to build a Bento with the latest
version of the `{{ cookiecutter.__full_name }}` model. This may take a while when running for the first
time for BentoML to resolve all dependency versions:

```
> bentoml build

[01:14:04 AM] INFO     Building BentoML service "{{ cookiecutter.__full_name }}:[[bento_tag]]" from build context      
                       "/home/chef/workspace/gallery/pytorch"                                                         
              INFO     Packing model "{{ cookiecutter.__project_slug }}:[[model_tag]]" from                               
                       "/home/chef/bentoml/models/{{ cookiecutter.__project_slug }}/[[model_tag]]"                       
              INFO     Locking PyPI package versions..                                                                 
[01:14:05 AM] INFO                                                                                                     
                       ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░                                   
                       ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░                                   
                       ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░                                   
                       ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░                                   
                       ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗                                   
                       ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝                                   
                                                                                                                       
              INFO     Successfully built Bento(tag="{{ cookiecutter.__full_name }}:[[bento_tag]]") at                 
                       "/home/chef/bentoml/bentos/{{ cookiecutter.__full_name }}/[[bento_tag]]/"                      
```

This Bento can now be loaded for serving:

```bash
bentoml serve {{ cookiecutter.__full_name }}:latest --production
```

The Bento directory contains all code, files, models and configs required for running this service.
BentoML standarlizes this file structure which enables serving runtimes and deployment tools to be
built on top of it. By default, Bentos are managed under the `~/bentoml/bentos` directory:

```
> cd ~/bentoml/bentos/{{ cookiecutter.__full_name }} && cd $(cat latest)

> tree
.
├── apis
│   └── openapi.yaml
├── bento.yaml
├── env
│   ├── conda
│   ├── docker
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   └── init.sh
│   └── python
│       ├── requirements.lock.txt
│       ├── requirements.txt
│       └── version.txt
├── models
│   └── {{ cookiecutter.__full_name }}
│       ├── [[model_tag]] 
│       │   ├── [[model_content]]
│       │   └── [[model_content]]
│       └── latest
├── README.md
└── src
    ├── model.py
    ├── service.py
    └── train.py

9 directories, 15 files
```


### Containerize Bento for deployment

Make sure you have docker installed and docker deamon running, and the following command
will use your local docker environment to build a new docker image, containing the model
server configured from this Bento:

```bash
bentoml containerize {{ cookiecutter.__full_name }}:latest
```

Test out the docker image built:
```bash
docker run -p 5000:5000 {{ cookiecutter.__full_name }}:[[docker_tag]]
```

# BentoML Gallery

This is a collection of machine learning projects that utilizes [BentoML](https://github.com/bentoml/BentoML)
for model serving. The goal is to demonstrate real-world BentoML usage and best practices in
productionizing a machine learning mdoel.


<!--TODO: Highlighted Projects-->


### How to run a gallery project

There are two ways to run each of the gallery projects:

* README.md - documents how to run the python scripts in this project directory
* {project_name}.ipynb - notebook shows the same process but all from a notebook environment


### Project layout

Each gallery project is under its own folder, containing the following files:

* README.md - background of the project and directions to run it
* {project_name}.ipynb - a jupyter notebook that shows the building process and run the project end to end
* requirements.txt - required PyPI packages for this project
* train.py - a python script for training an ML model and saving it with BentoML
* import_model.py - some projects might be importing existing models, this script helps import the model to BentoML
* service.py - a python script that defines the bentoml.Service instance for serving
* bentofile.yaml - the bento build file for building the service into a Bento
* .bentoignore - files to exclude from build directory, when building a Bento
* benchmark.py - a python script that tests the baseline performance of the final model server created


## How to contribute

If you have issues running these projects or have suggestions for improvement, use [Github Issues](https://github.com/bentoml/gallery/issues/new)

If you are interested in contributing new projects to this repo, make sure to do the following first:

* Join us on [Slack](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg) and share your idea in #dev channel
* Follow the basic structures of other existing gallery projects
* Ensure your project runs with the latest version of BentoML


workflow:
- add gallery root, run `make` -> generate a new cookiecutter-template

- edit `service.py`, `train.py`, `model.py`, `bentofile.yaml`, `requirements.txt`, the generated jupyter notebook

- edit the interactive code under the ipython shell section, check other example repo on how to edit it, and also remove all comments under this section

- edit README.md for the new projects:
    - [[endpoints]]: number of endpoints for the BentoML service definition under `service.py`
    - [[an_endpoint]]: one of the above endpoints for the curl example, modify headers and file object corespondingly
    - [[bento_tag]]: replace this with the example tags when run `bentoml build` for bento
    - [[model_tag]]: replace this with the example tags when run `bentoml build` for model
    - [[model_content]]: replace this with the content of this model under the Bento directory section (under the `tree` section)
    - [[docker_tag]]: when running `bentoml containerize` it should return a tag, replace that here

    the generated README will contain boilerplate code from the aforementioned
    files, so you will need to update those section under the readme

- then edit all the boilerplate files freely, make sure to document all additional steps or files in your tutorial.


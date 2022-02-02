#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import json

try:
    from cookiecutter.main import cookiecutter
except ImportError:
    raise ImportError("cookiecutter not found. Install cookiecutter with `pip install -U cookiecutter`")

NAME_PREFIX = "bentoml_gallery_"

gallery_root = Path(__file__).parent.parent
add_new_gallery_project_cookiecutter_root = gallery_root / "templates"

cookiecutter(str(add_new_gallery_project_cookiecutter_root))
directory = [d for d in  os.listdir(gallery_root) if NAME_PREFIX in d][0]

with open(directory + "/configuration.json", "r") as configuration_file:
    configuration = json.load(configuration_file)

project_dir = configuration['project_dir']
project_slug = configuration['project_slug']
notebook_filename = configuration['full_name']
include_tests = configuration['include_tests'] == "True"

result_gallery_dir = Path(gallery_root, project_dir, project_slug)
result_gallery_dir.mkdir(parents=True, exist_ok=True)

os.remove(str(Path(directory, 'configuration.json')))

shutil.move(str(Path(directory, "README.md")), str(Path(result_gallery_dir, "README.md")))
shutil.move(str(Path(directory, ".bentoignore")), str(Path(result_gallery_dir, ".bentoignore")))
shutil.move(str(Path(directory, f"{notebook_filename}_demo.ipynb")), str(Path(result_gallery_dir, f"{notebook_filename}_demo.ipynb")))
shutil.move(str(Path(directory, "service.py")), str(Path(result_gallery_dir, "service.py")))
shutil.move(str(Path(directory, "train.py")), str(Path(result_gallery_dir, "train.py")))
shutil.move(str(Path(directory, "model.py")), str(Path(result_gallery_dir, "model.py")))
shutil.move(str(Path(directory, "bentofile.yaml")), str(Path(result_gallery_dir, "bentofile.yaml")))
shutil.move(str(Path(directory, "requirements.txt")), str(Path(result_gallery_dir, "requirements.txt")))

if include_tests:
    shutil.move(str(Path(directory, "samples")), str(Path(result_gallery_dir, "samples")))
    shutil.move(str(Path(directory, "tests")), str(Path(result_gallery_dir, "tests")))

os.rmdir(directory)


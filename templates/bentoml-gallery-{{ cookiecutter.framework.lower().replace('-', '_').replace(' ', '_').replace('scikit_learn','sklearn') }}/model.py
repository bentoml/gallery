# Contributed by {{cookiecutter.author}}.
"""{{cookiecutter.framework}} {{cookiecutter.project_name}} model implementation."""

import {{ cookiecutter.framework.lower().replace(' ', '').replace('pytorch', 'torch') }}

class {{ cookiecutter.project_name.replace(' ','') }}():
    def __init__(self): ...

    def forward(self, x): ...

    def predict(self, inp): ...

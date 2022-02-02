project_slug = "bentoml_gallery_{{ cookiecutter.framework.lower().replace('-', '_').replace(' ', '_').replace('scikit_learn','sklearn') }}"

if hasattr(project_slug, "isidentifier"):
    assert (
        project_slug.isidentifier()
    ), f"'{project_slug}' project slug is not a valid Python identifier."

assert (
    project_slug == project_slug.lower()
), f"'{project_slug}' project slug should be all lowercase"

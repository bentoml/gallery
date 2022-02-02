.DEFAULT_GOAL := create-new-gallery-projects

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

create-new-gallery-projects: ## Create new gallery projects from templates
	@pip install -r ./scripts/requirements.txt
	@./scripts/gallery_cli.py

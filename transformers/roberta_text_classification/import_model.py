# Contributed by BentoML Team.
# Load a pretrained roberta pretrained model

import bentoml
import transformers
from bentoml.exceptions import NotFound
import logging
from config import MODEL_NAME, TASKS, MODEL

logger = logging.getLogger("bentoml")

if __name__ == "__main__":
    try:
        meta = bentoml.models.get(MODEL_NAME)
    except (NotFound, FileNotFoundError):
        logger.info(f"{MODEL_NAME} not found under BentoML modelstore, loading from HuggingFace...")
        classifier = transformers.pipeline(TASKS, model=MODEL, return_all_scores=True)
        meta = bentoml.transformers.save(MODEL_NAME, classifier)
    finally:
        logger.info(f"Tag found under BentoML modelstore: {meta.tag}\nYou can then import this model with:\n\tmodel = bentoml.transformers.load('{meta.tag}')")


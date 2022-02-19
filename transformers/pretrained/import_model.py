import logging

import bentoml
import transformers

TASKS = "text-classification"
BENTOML_MODEL_NAME = "roberta_text_classification"
MODEL = "j-hartmann/emotion-english-distilroberta-base"

logger = logging.getLogger("bentoml")
logging.captureWarnings(True)

classifier = transformers.pipeline(TASKS, model=MODEL, return_all_scores=True)
tag = bentoml.transformers.save(BENTOML_MODEL_NAME, classifier)
logger.info(f"Model saved under tag: {str(tag)}")

import bentoml

import transformers

TASKS = "text-classification"
BENTOML_MODEL_NAME = "roberta_text_classification"
MODEL = "j-hartmann/emotion-english-distilroberta-base"

classifier = transformers.pipeline(TASKS, model=MODEL, return_all_scores=True)  # type: ignore
tag = bentoml.transformers.save(BENTOML_MODEL_NAME, classifier)

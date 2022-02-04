# Contributed by BentoML Team.
# Transformers Roberta Text Classification model implementation.

import transformers
import bentoml

LM_HEAD = "sequence-classification"

tag = bentoml.transformers.import_from_huggingface_hub("j-hartmann/emotion-english-distilroberta-base", lm_head=LM_HEAD)

runner = bentoml.transformers.load_runner(tag, tasks='text-classification', lm_head=LM_HEAD, return_all_scores=True)

print(runner.run_batch(['I love you', "I don't want to spend time with you"]))

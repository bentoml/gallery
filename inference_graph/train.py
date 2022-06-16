import bentoml
import transformers


if __name__ == "__main__":
    # Create Transformers pipelines from pretrained models
    pipeline1 = transformers.pipeline(task="text-classification", model="bert-base-uncased", tokenizer="bert-base-uncased")
    pipeline2 = transformers.pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    pipeline3 = transformers.pipeline(task="text-classification", model="ProsusAI/finbert")

    # Save models to BentoML local model store
    bentoml.transformers.save_model("bert-base-uncased", pipeline1)
    bentoml.transformers.save_model("distilbert-base-uncased-finetuned-sst-2-english", pipeline2)
    bentoml.transformers.save_model("prosusai-finbert", pipeline3)

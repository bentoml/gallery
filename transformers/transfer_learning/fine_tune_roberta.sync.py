# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Transfer learning with Transformers 🤝 BentoML
#
# [Source](https://github.com/bentoml/gallery/tree/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb) | [nbviewer](https://nbviewer.org/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb) | [Colab](https://colab.research.google.com/github/bentoml/gallery/blob/main/transformers/roberta_text_classification/transfer_learning/fine_tune_roberta.sync.ipynb)
#
# In this Jupyter notebook file, we will do transfer learning a version of [distilroberta-base](https://huggingface.co/distilroberta-base) for emotion detection (sentiment analysis) from text.
#
# <b>Stack: Transformers(<i>PyTorch backend</i>) - BentoML </b>

# %% [markdown]
# ## Import pretrained model with BentoML
# Users can easily import the our fine tune model with `bentoml.models.import_model()`:

# %%
import bentoml
tag = bentoml.models.import_model("./exported")
model, tokenizer = bentoml.transformers.load(tag, return_config=False)

# %% [markdown]
# <b>NOTE:</b> If you just want to use the provided model, stop here and go back to [README.md](../README.md) for next step.


# %% [markdown]
# ## Fine-tuning for multi-class sentiment analysis in a different domain
# In this section, we will fine tune a version of [distilroberta-base](https://huggingface.co/distilroberta-base)

# %% [markdown]
# ### Install requirements

# %%
# !pip install -r requirements.txt

# %% [markdown]
# ### Setup pretrained model

# %%
# !python import_model.py

# %% [markdown]
# ### Sanity check

# %%
import transformers
import torch
import psutil

from config import (
    BENTOML_FINETUNE_NAME,
    MODEL,
    NUM_LABELS,
    NUM_EPOCHS,
    BATCH_SIZE,
    LR,
    WDECAY,
)
from transformers.trainer_utils import set_seed
from datasets.load import load_dataset

torch.set_num_threads(psutil.cpu_count())
set_seed(420)

# %% [markdown]
# ### Load Dataset
#
# We will use [emotion](https://huggingface.co/datasets/emotion) via [huggingface/datasets](https://huggingface.co/docs/datasets/)

# %%
emotion = load_dataset("emotion")

# %% [markdown]
# We will load tokenizer from imported model from HuggingFace hub.

# %%
with open("tags.log", "r", encoding="utf-8") as f:
    tag = f.readline().strip("\n")
_, tokenizer = bentoml.transformers.load(tag, return_config=False)

# %% [markdown]
# The following `preprocess_function` will [map](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map)
# all given text in the dataset to a tokenized version. We can then later use for training

# %%
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)


tokenized_emotion = emotion.map(preprocess_function, batched=True, batch_size=None)

# %% [markdown]
# We will use f1 and recall as our metrics for the model performance.

# %%
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# %%
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "./fine_tune/", num_labels=NUM_LABELS, ignore_mismatched_sizes=True
)

# %%
tokenized_emotion.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_emotion["train"].features

# %% [markdown]
# Setup training arguments as well as `transformers.Trainer`

# %%
logging_steps = len(tokenized_emotion["train"]) // BATCH_SIZE
training_args = transformers.TrainingArguments(
    output_dir="results",
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    weight_decay=WDECAY,
    save_strategy="no",
    evaluation_strategy="no",
    disable_tqdm=False,
)

# %%
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_emotion["train"],
    eval_dataset=tokenized_emotion["validation"],
)

# %% [markdown]
# Transformers provided an easy way to fine-tune given models with `Trainer` API. Simply do:

# %%
trainer_output = trainer.train()
trainer_output

# %% [markdown]
# ### Evaluate model performance.
# %%
results = trainer.evaluate()
results

# %% [markdown]
# ### Validation

# %%
preds_output = trainer.predict(tokenized_emotion["validation"])
preds_output.metrics

# %% [markdown]
# ### Save our fine-tune model to BentoML modelstore

# %%
metadata = results.update({"transfer-learning": True})
tag = bentoml.transformers.save(BENTOML_FINETUNE_NAME, model, tokenizer=tokenizer, metadata=metadata)
bentoml.models.export_model(tag, "./exported")

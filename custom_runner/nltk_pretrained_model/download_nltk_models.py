#!/usr/bin/env python
import os
import nltk

if os.environ.get("BENTO_PATH"):
    # During BentoML containerize
    download_dir = os.path.expandvars("/home/bentoml/nltk_data")
else:
    download_dir = os.path.expandvars("$HOME/nltk_data")

nltk.download('vader_lexicon', download_dir)
nltk.download('punkt', download_dir)

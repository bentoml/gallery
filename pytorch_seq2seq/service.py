import typing as t
import unicodedata
import re

import bentoml
from bentoml.io import Text


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


encoder = bentoml.pytorch.load_runner(
    "pytorch_tldr_encoder"
)

decoder = bentoml.pytorch.load_runner(
    "pytorch_tldr_decoder"
)

svc = bentoml.Service(
    name="pytorch_tldr_demo",
    runners=[
        encoder,
        decoder,
    ],
)


@svc.api(input=Text(), output=Text())
def predict(input_arr: t.List[str]) -> t.List[str]:
    input_arr = list(map(normalizeString, input_arr))
    enc_arr = encoder.run_batch(input_arr)
    res = decoder.run_batch(enc_arr)
    return res[0]['generated_text']

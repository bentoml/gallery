import typing as t
import unicodedata
import re
import pickle

import bentoml
from bentoml.io import Text

from train import evaluate


def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


encoder_runner = bentoml.pytorch.load_runner("pytorch_seq2seq_encoder:latest")
decoder_runner = bentoml.pytorch.load_runner("pytorch_seq2seq_decoder:latest")

svc = bentoml.Service(
    name="pytorch_seq2seq",
    runners=[
        encoder_runner,
        decoder_runner,
    ],
)


@svc.api(input=Text(), output=Text())
async def summarize(input_sentence: str) -> str:
    input_sentence = normalizeString(input_sentence)
    output, attn = await evaluate(encoder_runner, decoder_runner, input_sentence)
    output_sentence = " ".join(output)
    return output_sentence


@svc.api(input=Text(), output=Text())
async def summarize_batch(input_arr: [str]) -> [str]:
    input_arr = list(map(normalizeString, input_arr))
    results = []
    for input_sentence in input_arr:
        output, attn = await evaluate(encoder_runner, decoder_runner, input_sentence)
        results.append(" ".join(output))
    return results

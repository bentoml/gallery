import typing as t

import bentoml
from bentoml.io import Text


def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


encoder_runner = bentoml.pytorch.load("pytorch_seq2seq_encoder")
decoder_runner = bentoml.pytorch.load("pytorch_seq2seq_decoder")

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
    enc_sentence = await encoder.async_run(input_sentence)
    res = await decoder.async_run(enc_sentence)
    return res["generated_text"]


@svc.api(input=Text(), output=Text())
async def summarize_batch(input_arr: [str]) -> [str]:
    input_arr = list(map(normalizeString, input_arr))
    results = []
    for input_sentence in input_arr:
        enc_arr = await encoder.async_run(input_arr)
        res = await decoder.async_run(enc_arr)
        results.append(res["generated_text"])
    return results

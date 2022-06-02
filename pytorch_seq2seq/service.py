import typing as t
import unicodedata
import re
import pickle

import bentoml
from bentoml.io import Text
from bentoml.pytorch import PytorchRunner
# from bentoml.models? import import_model, export_model

# from train import evaluate


class Seq2Seq(PytorchRunner):

    def __init__(self, encoder_tag, decoder_tag):
        self.encoder_tag = encoder_tag
        self.decoder_tag = decoder_tag

        # self.vectors = import_model("seq2seq_vectors.pkl") ?
        self.vectors = {}

        self.max_length = 260
        super().__init__()

    def _setup(self):
        self.encoder = bentoml.models.get(self.encoder_tag)
        self.decoder = bentoml.models.get(self.decoder_tag)
        super()._setup()

    def _run_batch(self, *args):
        encodings = []
        for arg in args:
            encodings.append(' '.join(self._evaluate(arg)))
        return encodings[0]

    def _evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = tensorFromSentence(
                self.vectors["input_lang"], sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(
                self.max_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append("<EOS>")
                    break
                else:
                    decoded_words.append(
                        self.vectors["output_lang"].index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[: di + 1]


def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


seq2seq_runner = Seq2Seq("pytorch_seq2seq_encoder:latest",
                         "pytorch_seq2seq_decoder:latest")

svc = bentoml.Service(
    name="pytorch_seq2seq",
    runners=[
        seq2seq_runner
    ],
)


@svc.api(input=Text(), output=Text())
async def summarize(input_sentence: str) -> str:
    input_sentence = normalizeString(input_sentence)
    return seq2seq_runner.run(input_sentence)


@svc.api(input=Text(), output=Text())
async def summarize_batch(input_arr: [str]) -> [str]:
    input_arr = list(map(normalizeString, input_arr))
    return seq2seq_runner.run(input_sentence)

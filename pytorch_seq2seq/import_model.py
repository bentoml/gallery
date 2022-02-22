import logging

import bentoml
import torch
from model import EncoderRNN, DecoderRNN, AttnDecoderRNN


MODEL = 'seq2seq.pt'
BENTOML_MODEL_NAME = "pytorch_seq2seq"


if __name__ == "__main__":
    logger = logging.getLogger("bentoml")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(MODEL, map_location=device)
    tag = bentoml.pytorch.save(BENTOML_MODEL_NAME, MODEL)
    logger.info(f'Model saved under tag: {str(tag)}')

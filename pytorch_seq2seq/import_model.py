import logging

import bentoml
import torch
from model import EncoderRNN, DecoderRNN, AttnDecoderRNN


MODEL = "../models/seq2seq.pt"
BENTOML_MODEL_NAME = "pytorch_seq2seq_"


if __name__ == "__main__":
    logger = logging.getLogger("bentoml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL, map_location=device)

    encoder_tag = bentoml.pytorch.save(BENTOML_MODEL_NAME + "encoder", MODEL["encoder"])
    decoder_tag = bentoml.pytorch.save(BENTOML_MODEL_NAME + "decoder", MODEL["decoder"])

    logger.info(f"Model saved under tag: {str(encoder_tag)}")
    logger.info(f"Model saved under tag: {str(decoder_tag)}")

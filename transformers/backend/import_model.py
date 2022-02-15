import bentoml
import transformers
import logging

from pathlib import Path
from bentoml._internal.models import Model
from config import BENTOML_MODEL_NAME, TASKS, MODEL

logger = logging.getLogger("bentoml")

FILE = Path(__file__).parent
TAGFILE = Path(FILE, "tags.log")

if __name__ == "__main__":
    if not TAGFILE.absolute().exists():
        logger.info(f"{TAGFILE.name} does not exists in current directory.")
        meta = None
        logger.info(
            f"Checking if {BENTOML_MODEL_NAME} already exists under BentoML modelstore..."
        )
        try:
            meta = bentoml.models.get(BENTOML_MODEL_NAME)
        except Exception:
            logger.info(
                f"{BENTOML_MODEL_NAME} not found under BentoML modelstore, loading from HuggingFace from pipeline({TASKS})..."
            )
            classifier = transformers.pipeline(
                TASKS, model=MODEL, return_all_scores=True
            )
            meta = bentoml.transformers.save(BENTOML_MODEL_NAME, classifier)
        finally:
            assert meta is not None
            tag = meta if not isinstance(meta, Model) else meta.tag
            logger.info(f"Tag under BentoML modelstore: {tag}. Saving to {TAGFILE}...")
            with open(TAGFILE, "w", encoding="utf-8") as f:
                f.write(f"{str(tag)}")
            f.close()
    else:
        logger.info(
            f"{TAGFILE.name} exists under this directory. This means you have already run this script once. Since"
            f" we are dealing with pretrained model, we don't want to re-download it multiple times. Returning previous tag from {TAGFILE.name}"
        )
        with open(TAGFILE, "r", encoding="utf-8") as f:
            tag = f.readline()
            logger.info(f"model tag: {tag}")
        f.close()

import numpy as np
import PIL.Image

import bentoml
from bentoml.io import Image
from bentoml.io import JSON


yolo_runner = bentoml.pytorch.get("yolo").to_runner()

svc = bentoml.Service(
    name="pytorch_yolo_demo",
    runners=[yolo_runner],
)


@svc.api(input=Image(), output=JSON())
async def predict_image(img: PIL.Image.Image) -> list:
    assert isinstance(img, PIL.Image.Image)
    return await yolo_runner.async_run([np.array(img)])


import imageio
import numpy as np

from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import FastaiModelArtifact
from bentoml.adapters import FastaiImageInput

from fastai.vision import *
from fastai.vision import Image, pil2tensor

@env(pip_dependencies=['fastai'])
@artifacts([FastaiModelArtifact('pet_classifer')])
class PetClassification(BentoService):
    
    @api(input=FastaiImageInput())
    def predict(self, image):
        fastai_image = pil2tensor(image, np.float32)
        fastai_image = Image(fastai_image)
        result = self.artifacts.pet_classifer.predict(fastai_image)
        return str(result)

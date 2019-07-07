
from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import FastaiModelArtifact
from bentoml.handlers import ImageHandler

@env(conda_environment=['fastai'])
@artifacts([FastaiModelArtifact('learner')])
class PetClassification(BentoService):
    
    @api(ImageHandler, fastai_model=True)
    def predict(self, image):
        result = self.artifacts.learner.predict(image)
        return str(result)

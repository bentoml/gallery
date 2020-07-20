
from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import FastaiModelArtifact
from bentoml.adapters import FastaiImageInput

@env(pip_dependencies=['fastai'])
@artifacts([FastaiModelArtifact('pet_classifer')])
class PetClassification(BentoService):
    
    @api(input=FastaiImageInput())
    def predict(self, image):
        result = self.artifacts.pet_classifer.predict(image)
        return str(result)

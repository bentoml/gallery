
import numpy as np
from PIL import Image
from bentoml import api, artifacts, env, BentoService
from bentoml.artifact import KerasModelArtifact
from bentoml.adapters import ImageInput

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@env(pip_dependencies=['keras==2.3.1', 'tensorflow==1.14.0', 'pillow', 'numpy'])
@artifacts([KerasModelArtifact('classifier')])
class KerasFashionMnistService(BentoService):
        
    @api(input=ImageInput(pilmode='L'))
    def predict(self, imgs):
        inputs = []
        for img in imgs:
            img = Image.fromarray(img).resize((28, 28))
            img = np.array(img.getdata()).reshape((28, 28, 1))
            inputs.append(img)
        inputs = np.stack(inputs)
        class_idxs = self.artifacts.classifier.predict_classes(inputs)
        return [class_names[class_idx] for class_idx in class_idxs]

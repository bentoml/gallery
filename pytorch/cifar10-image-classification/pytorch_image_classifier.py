
from PIL import Image

from torch.autograd import Variable
from torchvision import transforms

import bentoml
from bentoml.artifact import PytorchModelArtifact
from bentoml.handlers import ImageHandler


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

@bentoml.env(pip_dependencies=['torch', 'numpy', 'torchvision', 'scikit-learn'])
@bentoml.artifacts([PytorchModelArtifact('net')])
class ImageClassifier(bentoml.BentoService):
    @bentoml.api(ImageHandler)
    def predict(self, img):
        img = Image.fromarray(img).resize((32, 32))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        input_data = transform(img)

        outputs = self.artifacts.net(Variable(input_data).unsqueeze(0))
        _, output_classes = outputs.max(dim=1)

        return classes[output_classes]

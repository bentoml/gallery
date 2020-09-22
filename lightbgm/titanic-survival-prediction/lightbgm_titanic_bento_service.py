
import lightgbm as lgb

import bentoml
from bentoml.frameworks.lightgbm import LightGBMModelArtifact
from bentoml.adapters import DataframeInput

@bentoml.artifacts([LightGBMModelArtifact('model')])
@bentoml.env(pip_packages=['lightgbm'])
class TitanicSurvivalPredictionService(bentoml.BentoService):
    
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        data = df[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']]
        return self.artifacts.model.predict(data)

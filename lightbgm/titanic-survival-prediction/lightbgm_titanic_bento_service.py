
import lightgbm as lgb

import bentoml
from bentoml.artifact import LightGBMModelArtifact
from bentoml.handlers import DataframeHandler

@bentoml.artifacts([LightGBMModelArtifact('model')])
@bentoml.env(pip_dependencies=['lightgbm'])
class TitanicSurvivalPredictionService(bentoml.BentoService):
    
    @bentoml.api(DataframeHandler)
    def predict(self, df):
        data = df[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']]
        return self.artifacts.model.predict(data)

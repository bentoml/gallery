
import xgboost as xgb

import bentoml
from bentoml.artifact import XgboostModelArtifact
from bentoml.adapters import DataframeInput

@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([XgboostModelArtifact('model')])
class TitanicSurvivalPredictionXgBoost(bentoml.BentoService):
    
    @bentoml.api(input=DataframeInput())
    def predict(self, df):
        data = xgb.DMatrix(data=df[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']])
        return self.artifacts.model.predict(data)

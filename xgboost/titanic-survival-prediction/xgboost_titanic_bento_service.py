
import xgboost as xgb

import bentoml
from bentoml.frameworks.xgboost import XgboostModelArtifact
from bentoml.adapters import DataframeInput

@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([XgboostModelArtifact('model')])
class TitanicSurvivalPredictionXgBoost(bentoml.BentoService):
    
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        data = xgb.DMatrix(data=df[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']])
        return self.artifacts.model.predict(data)


from bentoml import api, env, BentoService, artifacts
from bentoml.frameworks.xgboost import XgboostModelArtifact
from bentoml.adapters import DataframeInput

import xgboost as xgb

@env(pip_packages=['xgboost'])
@artifacts([XgboostModelArtifact('model')])
class LeagueWinPrediction(BentoService):
    
    @api(input=DataframeInput(), batch=True)
    def predict(self, df):
        dmatrix = xgb.DMatrix(df)
        return self.artifacts.model.predict(dmatrix)

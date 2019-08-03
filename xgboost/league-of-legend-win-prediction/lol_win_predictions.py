
from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import XgboostModelArtifact
from bentoml.handlers import DataframeHandler

import xgboost as xgb

@env(pip_dependencies=['xgboost'])
@artifacts([XgboostModelArtifact('model')])
class LeagueWinPrediction(BentoService):
    
    @api(DataframeHandler)
    def predict(self, df):
        dmatrix = xgb.DMatrix(df)
        return self.artifacts.model.predict(dmatrix)

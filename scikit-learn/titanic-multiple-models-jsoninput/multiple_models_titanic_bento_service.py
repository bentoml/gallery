
import json

import bentoml
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from bentoml.adapters import JsonInput
from bentoml.artifact import SklearnModelArtifact


@bentoml.artifacts([SklearnModelArtifact("xgb"), SklearnModelArtifact("lgb")])
@bentoml.env(
    conda_channels=["conda-forge"],
    conda_dependencies=["lightgbm==2.3.*", "pandas==1.0.*", "xgboost==1.2.*"],
)
class TitanicSurvivalPredictionService(bentoml.BentoService):
    @bentoml.api(input=JsonInput())
    def predict(self, datain):
        # datain is a list of a json object.
        df = pd.read_json(json.dumps(datain[0]), orient="table")

        data = df[["Pclass", "Age", "Fare", "SibSp", "Parch"]]
        result = pd.DataFrame()
        result["xgb_proba"] = self.artifacts.xgb.predict_proba(data)[:, 1]
        result["lgb_proba"] = self.artifacts.lgb.predict_proba(data)[:, 1]
        # make sure to return as a list of json
        return [result.to_json(orient="table")]

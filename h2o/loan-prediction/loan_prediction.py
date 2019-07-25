
from bentoml import api, env, artifacts, BentoService
from bentoml.artifact import H2oModelArtifact
from bentoml.handlers import DataframeHandler

import h2o

@env(conda_pip_dependencies = ['h2o', 'pandas'])
@artifacts([H2oModelArtifact('model')])
class LoanPrediction(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        h2o_frame = h2o.H2OFrame(df)
        return self.artifacts.model.predict(h2o_frame)

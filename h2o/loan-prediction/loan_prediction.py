
import h2o

from bentoml import api, env, artifacts, BentoService
from bentoml.artifact import H2oModelArtifact
from bentoml.adapters import DataframeInput

@env(pip_dependencies = ['h2o', 'pandas'])
@artifacts([H2oModelArtifact('model')])
class LoanPrediction(BentoService):
    
    @api(input=DataframeInput)
    def predict(self, df):
        h2o_frame = h2o.H2OFrame(df, na_strings=['NaN'])
        predictions = self.artifacts.model.predict(h2o_frame)
        return predictions.as_data_frame()

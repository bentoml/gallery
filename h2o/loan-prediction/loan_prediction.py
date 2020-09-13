
import h2o

from bentoml import api, env, artifacts, BentoService
from bentoml.frameworks.h2o import H2oModelArtifact
from bentoml.adapters import DataframeInput

@env(
    pip_dependencies=['h2o==3.24.0.2', 'pandas'],
    conda_channels=['h2oai'],
    conda_dependencies=['h2o==3.24.0.2']
)
@artifacts([H2oModelArtifact('model')])
class LoanPrediction(BentoService):
    
    @api(input=DataframeInput())
    def predict(self, df):
        h2o_frame = h2o.H2OFrame(df, na_strings=['NaN'])
        predictions = self.artifacts.model.predict(h2o_frame)
        return predictions.as_data_frame()


from bentoml import api, BentoService, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler

@env(pip_dependencies=['sklearn', 'numpy', 'pandas'])
@artifacts([PickleArtifact('grid')])
class LoanPrediction(BentoService):
    
    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.grid.predict(df)

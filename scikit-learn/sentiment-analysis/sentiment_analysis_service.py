import pandas as pd
import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler

@bentoml.artifacts([PickleArtifact('model')])
@bentoml.env(pip_dependencies=["scikit-learn", "pandas"])
class SentimentAnalysisService(bentoml.BentoService):

    @bentoml.api(DataframeHandler, typ='series')
    def predict(self, series):
        """
        predict expects pandas.Series as input
        """        
        return self.artifacts.model.predict(series)

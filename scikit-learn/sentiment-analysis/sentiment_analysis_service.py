import pandas as pd
import bentoml
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.handlers import DataframeHandler
from bentoml.adapters import DataframeInput

@bentoml.artifacts([PickleArtifact('model')])
@bentoml.env(pip_packages=["scikit-learn", "pandas"])
class SKSentimentAnalysis(bentoml.BentoService):

    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        """
        predict expects pandas.Series as input
        """        
        series = df.iloc[0,:]
        return self.artifacts.model.predict(series)

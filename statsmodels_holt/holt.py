
# holt.py
from bentoml import env, artifacts, api, BentoService
from bentoml.handlers import DataframeHandler,DataframeInput
from bentoml.artifact import PickleArtifact
import numpy as np

@env(pip_dependencies=["statsmodels==0.10.1","joblib","numpy"],conda_dependencies=["ruamel.yaml==0.16"])

@artifacts([PickleArtifact('model')])
class holt(BentoService):
    @api(input=DataframeInput())
    def predict(self, df):

        # Printing the dataframe to cross-cjheck
        print(df.head())

        # Parsing the dataframe values
        weeks=int(df.iat[0,0])
        print(weeks)
        return((self.artifacts.model).forecast(weeks))
  

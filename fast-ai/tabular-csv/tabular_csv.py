
from bentoml import env, api, artifacts, BentoService
from bentoml.artifact import FastaiModelArtifact
from bentoml.handlers import DataframeHandler


@env(conda_environment=['fastai'])
@artifacts([FastaiModelArtifact('model')])
class TabularModel(BentoService):
    
    @api(DataframeHandler)
    def predict(self, df):
        result = []
        for index, row in df.iterrows():            
            result.append(self.artifacts.model.predict(row))
        return str(result)

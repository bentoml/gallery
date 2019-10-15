
from bentoml import env, api, artifacts, BentoService
from bentoml.artifact import FastaiModelArtifact
from bentoml.handlers import DataframeHandler


@env(pip_dependencies=['fastai'])
@artifacts([FastaiModelArtifact('model')])
class TabularModel(BentoService):
    
    @api(DataframeHandler)
    def predict(self, df):
        results = []
        for _, row in df.iterrows():       
            prediction = self.artifacts.model.predict(row)
            results.append(prediction[0].obj)
        return results

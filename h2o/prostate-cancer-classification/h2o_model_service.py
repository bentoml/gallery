import pandas as pd
import h2o
import bentoml
from bentoml.frameworks.h2o import H2oModelArtifact
from bentoml.handlers import DataframeHandler

@bentoml.artifacts([H2oModelArtifact('model')])
@bentoml.env(
    pip_packages=['pandas', 'h2o==3.24.0.2'],
    conda_channels=['h2oai'],
    conda_dependencies=['h2o==3.24.0.2']
)
class H2oModelService(bentoml.BentoService):

    @bentoml.api(DataframeHandler)
    def predict(self, df):     
        hf = h2o.H2OFrame(df)
        predictions = self.artifacts.model.predict(hf)
        return predictions.as_data_frame()

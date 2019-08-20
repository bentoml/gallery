
from bentoml import api, artifacts, env, BentoService
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler

from keras.preprocessing import text, sequence
import numpy as np

list_of_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
max_text_length = 400

@env(conda_pip_dependencies=['keras', 'pandas', 'numpy'])
@artifacts([PickleArtifact('x_tokenizer'), PickleArtifact('model')])
class ToxicCommentClassification(BentoService):
    def tokenize_df(self, df):
        tokenized = self.artifacts.x_tokenizer.texts_to_sequences(df)
        return sequence.pad_sequences(tokenized, maxlen=max_text_length)
    
    @api(DataframeHandler)
    def predict(self, df):
        input_data = self.tokenize_df(df)
        prediction_result = self.artifacts.model.predict(input_data)
        result_data = np.where(prediction_result == np.amax(prediction_result))
        result = list_of_classes[result_data]
        return result
        


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
        comments = df['comment_text'].values
        tokenized = self.artifacts.x_tokenizer.texts_to_sequences(comments)        
        input_data = sequence.pad_sequences(tokenized, maxlen=max_text_length)
        return input_data
    
    @api(DataframeHandler)
    def predict(self, df):
        input_data = self.tokenize_df(df)
        prediction = self.artifacts.model.predict(input_data)
        result = []
        for i in prediction:
            result.append(list_of_classes[np.argmax(i)])
        return result

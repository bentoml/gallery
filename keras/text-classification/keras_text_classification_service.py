import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence, text
from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import KerasModelArtifact, PickleArtifact
from bentoml.adapters import JsonInput


max_features = 1000

@artifacts([
    KerasModelArtifact('model'),
    PickleArtifact('word_index')
])
@env(pip_dependencies=['tensorflow==1.14.0', 'numpy', 'pandas'])
class KerasTextClassificationService(BentoService):
   
    def word_to_index(self, word):
        if word in self.artifacts.word_index and self.artifacts.word_index[word] <= max_features:
            return self.artifacts.word_index[word]
        else:
            return self.artifacts.word_index["<UNK>"]
    
    def preprocessing(self, text_str):
        sequence = text.text_to_word_sequence(text_str)
        return list(map(self.word_to_index, sequence))
    
    @api(input=JsonInput())
    def predict(self, parsed_jsons):
        input_datas = [self.preprocessing(parsed_json['text']) for parsed_json in parsed_jsons]
        input_datas = sequence.pad_sequences(input_datas,
                                            value=self.artifacts.word_index["<PAD>"],
                                            padding='post',
                                            maxlen=80)

        return self.artifacts.model.predict_classes(input_datas).T[0]

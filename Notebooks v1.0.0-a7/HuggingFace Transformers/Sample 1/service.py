
import bentoml
from bentoml.io import Text

model_tag = "translation:latest"

translation_runner = bentoml.transformers.load_runner(model_tag,tasks='translation')

translate = bentoml.Service("translation", runners=[translation_runner])

@translate.api(input=Text(), output=Text())
def translate_text(input_series: str) -> str:
    try:
        result = translation_runner.run(input_series)
        print(result)
        return result['translation_text']
    except:
        return 'Invalid Input'


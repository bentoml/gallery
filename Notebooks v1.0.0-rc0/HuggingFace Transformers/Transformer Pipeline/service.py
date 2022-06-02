
import bentoml
from bentoml.io import Text

model_tag = "summarization_rc0:latest"

summarize_runner = bentoml.transformers.get(model_tag).to_runner()
summarize_model = bentoml.models.get(model_tag)

summarize = bentoml.Service("summarization_rc0", runners=[summarize_runner])


@summarize.api(input=Text(), output=Text())
def summarize_text(input_series: str) -> str:
    try:
        result = summarize_runner.run(input_series)
        return result[0]['summary_text']
    except:
        return 'Invalid Input'

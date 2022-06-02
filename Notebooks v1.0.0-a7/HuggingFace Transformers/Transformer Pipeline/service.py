
import bentoml
from bentoml.io import Text

model_tag = "summarization:latest"

summarize_runner = bentoml.transformers.load_runner(model_tag,tasks='summarization')
summarize_model = bentoml.models.get(model_tag)

summarize = bentoml.Service("summarization", runners=[summarize_runner])


@summarize.api(input=Text(), output=Text())
def summarize_text(input_series: str) -> str:
    try:
        result = summarize_runner.run(input_series)
        return result['summary_text']
    except:
        return 'Invalid Input'

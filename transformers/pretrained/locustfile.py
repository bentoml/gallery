import json
import random

from locust import task
from locust import between
from locust import HttpUser
from essential_generators import DocumentGenerator

random.seed(400)
gen = DocumentGenerator()

BATCH = 5
INPUTS = [gen.sentence() for _ in range(BATCH)]


class PreTrainedRoberta(HttpUser):

    wait_time = between(0.01, 2)

    @task
    def sentiment(self):
        self.client.post(
            "/sentiment", "I hope we will meet one day, but now, I have to leave."
        )

    @task
    def batch_sentiment(self):
        self.client.post("/batch_sentiment", json.dumps({"text": INPUTS}))

    @task
    def batch_all_scores(self):
        self.client.post("/batch_all_scores", json.dumps({"text": INPUTS}))

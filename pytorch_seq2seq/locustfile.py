from locust import task
from locust import between
from locust import HttpUser

test_text = 'Hello, world!'


class Seq2Seq(HttpUser):

    def __init__(self):
        self.wait_time = between(0.01, 2)

    @task
    def all_scores(self):
        self.client.post("/all_scores", test_text)

from locust import task
from locust import between
from locust import HttpUser

test_text = "I love you"


class FinetuneRoberta(HttpUser):

    wait_time = between(0.01, 2)

    @task
    def compare(self):
        self.client.post("/compare", test_text)

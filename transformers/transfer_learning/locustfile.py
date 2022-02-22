from locust import task
from locust import between
from locust import HttpUser

test_text = '["I love you", "I hope we will meet one day, but now, I have to leave."]'


class PreTrainedRoberta(HttpUser):

    wait_time = between(0.01, 2)

    @task
    def sentiment(self):
        self.client.post("/sentiment", test_text)

    @task
    def all_scores(self):
        self.client.post("/all_scores", test_text)

from locust import task
from locust import between
from locust import HttpUser

from essential_generators import MarkovTextGenerator

from model import MAX_LENGTH


class Seq2Seq(HttpUser):
    def __init__(self):
        super().__init__()  # Initialize the HttpUser class
        self.generator = MarkovTextGenerator()
        self.wait_time = between(0.01, 1)

    @task
    def summarize(self):
        self.client.post("/summarize", self.generator.gen_text(max_len=MAX_LENGTH))

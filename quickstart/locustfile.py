import numpy as np
from sklearn import datasets
from locust import HttpUser, task, between

test_data = datasets.load_iris().data
num_of_rows = test_data.shape[0]


class IrisLoadTestUser(HttpUser):

    @task
    def classify(self):
        input_data = test_data[np.random.choice(num_of_rows)]
        self.client.post("/classify", json=list(input_data))

    wait_time = between(0.01, 2)

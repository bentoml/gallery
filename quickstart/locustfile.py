import numpy as np
from sklearn import datasets
from locust import HttpUser, task, between

test_data = datasets.load_iris().data
num_of_rows = test_data.shape[0]
max_batch_size = 10


class IrisLoadTestUser(HttpUser):

    @task
    def classify(self):
        start = np.random.choice(num_of_rows - max_batch_size)
        end = start + np.random.choice(max_batch_size)

        input_data = test_data[start:end]
        self.client.post("/classify", json=input_data.tolist())

    wait_time = between(0.01, 2)

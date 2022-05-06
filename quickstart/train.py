from sklearn import svm, datasets

import bentoml


if __name__ == "__main__":
    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # Model Training
    clf = svm.SVC()
    clf.fit(X, y)

    # Save model to BentoML local model store
    bentoml.sklearn.save_model("iris_clf", clf)

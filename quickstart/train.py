from sklearn import svm, datasets

import bentoml


if __name__ == "__main__":
    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # Model Training
    clf = svm.SVC()
    clf.fit(X, y)

    bentoml.sklearn.save("iris_clf", clf)

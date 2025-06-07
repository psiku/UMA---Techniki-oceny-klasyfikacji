class AbstractClassifier:
    def __init__(self, model, x_train, x_test, y_train):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        return self.model.predict(self.x_test)

    def predict_proba(self):
        return self.model.predict_proba(self.x_test)

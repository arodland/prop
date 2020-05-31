import numpy as np

class LogSpace:
    def __init__(self, base_model):
        self.base_model = base_model

    def train(self, df, t):
        self.base_model.train(df, np.log(t))

    def predict(self, latitude, longitude):
        return np.exp(self.base_model.predict(latitude, longitude))

class Product:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def train(self, df, t1, t2):
        self.model1.train(df, t1)
        self.model2.train(df, t2)

    def predict(self, latitude, longitude):
        model1_pred = self.model1.predict(latitude, longitude)
        model2_pred = self.model2.predict(latitude, longitude)
        return model1_pred * model2_pred

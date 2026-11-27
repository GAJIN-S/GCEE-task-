from sklearn.linear_model import LogisticRegression


def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


def predict(model, X):
    return model.predict(X)


def predict_probability(model, X):
    return model.predict_proba(X)
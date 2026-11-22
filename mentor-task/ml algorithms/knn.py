from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def train_model(X_train, y_train):

    model = KNeighborsClassifier(
        n_neighbors=3
    )

    model.fit(X_train, y_train)

    return model


def predict(model, X_test):

    return model.predict(X_test)


def predict_probability(model, X_test):

    return model.predict_proba(X_test)


def evaluate_model(y_test, predictions):

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions
    )

    return accuracy, matrix, report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def train_model(X_train, y_train):
    model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=2,  # Minimum samples required to split a node
    min_samples_leaf=1,   # Minimum samples required at a leaf node
    criterion="gini",
    random_state=42
)

    model.fit(X_train, y_train)

    return model


def predict(model, X_test):
    return model.predict(X_test)


def predict_probability(model, X_test):
    return model.predict_proba(X_test)


def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, matrix, report
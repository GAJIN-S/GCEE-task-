from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
# 1. Load dataset

iris = load_iris()

X = iris.data
y = iris.target

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
activations = ["identity", "logistic", "tanh", "relu"]
for activation in activations:
    
    model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=2000,
        random_state=42
    )
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        zero_division=0
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# 7. Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nPredicted flower:", iris.target_names[prediction[0]])

# visulaisation 

nn=model.named_steps["mlpclassifier"]
print("Iterations completed:", nn.n_iter_)
print("Final training loss:", nn.loss_)

# Plot the loss curve
plt.plot(nn.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Neural Network Learning Curve")
plt.grid(True)
plt.show()
# 3. Create the neural network

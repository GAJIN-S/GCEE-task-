
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# STEP 1: Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print("Classes:", data.target_names)

# STEP 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 3: Choose activation functions
activations = ["identity", "logistic", "tanh", "relu"]

results = {}
trained_models = {}

# STEP 4: Train a separate model for each activation
for activation in activations:

    print("\nActivation:", activation)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation=activation,
            solver="adam",
            learning_rate_init=0.001,
            max_iter=1500,
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    # STEP 5: Predict test data
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    nn = model.named_steps["mlpclassifier"]

    results[activation] = accuracy
    trained_models[activation] = model

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Iterations:", nn.n_iter_)
    print("Final training loss:", round(nn.loss_, 5))

# STEP 6: Compare accuracy
print("\n--- Accuracy Comparison ---")

for activation, accuracy in results.items():
    print(f"{activation:10s}: {accuracy * 100:.2f}%")

# STEP 7: Plot accuracy
plt.figure(figsize=(8, 5))

plt.bar(results.keys(), results.values())
plt.ylabel("Test Accuracy")
plt.xlabel("Activation Function")
plt.title("Activation Function Comparison")
plt.ylim(0, 1.05)
plt.grid(axis="y")
plt.show()

# STEP 8: Plot training loss curves
plt.figure(figsize=(9, 5))

for activation, model in trained_models.items():
    nn = model.named_steps["mlpclassifier"]
    plt.plot(nn.loss_curve_, label=activation)

plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Loss Curves for Different Activations")
plt.legend()
plt.grid(True)
plt.show()

# STEP 9: Inspect one model in detail
best_activation = max(results, key=results.get)
best_model = trained_models[best_activation]

print("\nDetailed report for:", best_activation)

print(
    classification_report(
        y_test,
        best_model.predict(X_test),
        target_names=data.target_names
    )
)

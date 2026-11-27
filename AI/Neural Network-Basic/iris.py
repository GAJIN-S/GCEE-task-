from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()

# Convert features into a table
X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Target labels
y = iris.target

print("Dataset shape:", X.shape)
print("\nFirst five rows:")
print(X.head())

print("\nFlower species:")
print(iris.target_names)

print("\nFirst five labels:")
print(y[:5])

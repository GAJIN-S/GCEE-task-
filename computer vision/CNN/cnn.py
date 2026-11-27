import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

# 1. Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)

# 2. Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 3. Add the channel dimension
# Shape changes from (28, 28) to (28, 28, 1)
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

# 4. Build the CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Learn image features
    layers.Conv2D(
        32, kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    # Reduce spatial dimensions
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Learn more complex features
    layers.Conv2D(
        64, kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convert feature maps into a vector
    layers.Flatten(),

    # Learn combinations of features
    layers.Dense(64, activation="relu"),

    # Predict one of 10 digits
    layers.Dense(10, activation="softmax")
])

# 5. Configure the training process
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train the CNN
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# 7. Evaluate on unseen test images
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# 8. Predict one test image
index = 0

probabilities = model.predict(
    X_test[index:index + 1],
    verbose=0
)[0]

predicted_digit = probabilities.argmax()

plt.imshow(X_test[index].squeeze(), cmap="gray")
plt.title(
    f"Predicted: {predicted_digit} | "
    f"Actual: {y_test[index]}"
)

plt.axis("off")
plt.show()


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Configuration
vocab_size = 10000
max_length = 200

# 2. Load the IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("First review as token IDs:", X_train[0][:20])
print("Sentiment label:", y_train[0])

# 3. Make all reviews the same length
X_train = pad_sequences(
    X_train,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

# 4. Build the RNN
model = models.Sequential([
    layers.Input(shape=(max_length,)),

    # Convert word IDs into learned vectors
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=32,
        mask_zero=True
    ),

    # Process the sequence one step at a time
    layers.GRU(32),

    # Binary sentiment prediction
    layers.Dense(1, activation="sigmoid")
])

# 5. Configure training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# 7. Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# 8. Predict one test review
probability = model.predict(
    X_test[:1], verbose=0
)[0][0]

sentiment = "Positive" if probability >= 0.5 else "Negative"

print("Predicted sentiment:", sentiment)
print("Positive-class probability:", round(float(probability), 4))
print("Actual label:", int(y_test[0]))

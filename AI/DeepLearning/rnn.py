
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
vocab_size = 10000
max_length = 200

# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

# Padding
X_train = pad_sequences(X_train, maxlen=max_length, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")

# Build RNN model
model = models.Sequential([
    layers.Input(shape=(max_length,)),
    layers.Embedding(vocab_size, 32, mask_zero=True),
    layers.SimpleRNN(32),
    layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("RNN Test Loss:", round(loss, 4))
print("RNN Test Accuracy:", round(accuracy, 4))

import tensorflow as tf
from tensorflow import keras
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# Load the MNIST dataset from Hugging Face
dataset = load_dataset("mnist")
train_data = dataset["train"]
test_data = dataset["test"]

# Preprocess the data
def preprocess_data(data):
    images = np.array([np.array(img) for img in data["image"]])
    labels = np.array(data["label"])
    images = images.reshape((-1, 28, 28, 1)) / 255.0
    return images, labels

train_images, train_labels = preprocess_data(train_data)
test_images, test_labels = preprocess_data(test_data)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(128)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
line1, = ax1.plot([], [], label="Training Loss")
line2, = ax2.plot([], [], label="Training Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.legend()
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training Accuracy")
ax2.legend()

# Update the plot in real-time
def update_plot(epoch, losses, accuracies):
    line1.set_data(range(1, epoch + 2), losses)
    line2.set_data(range(1, epoch + 2), accuracies)
    ax1.set_xlim(1, epoch + 1)
    ax1.set_ylim(0, max(losses) + 0.1)
    ax2.set_xlim(1, epoch + 1)
    ax2.set_ylim(0, 1)
    display.clear_output(wait=True)
    display.display(fig)
    fig.canvas.flush_events()

# Train the neural network
num_epochs = 10
batch_size = 128

losses = []
accuracies = []

for epoch in range(num_epochs):
    # Train the model for one epoch
    history = model.fit(train_dataset, epochs=1, verbose=0)
    
    # Get the loss and accuracy for the current epoch
    loss = history.history["loss"][0]
    accuracy = history.history["accuracy"][0]
    losses.append(loss)
    accuracies.append(accuracy)
    
    # Update the plot
    update_plot(epoch, losses, accuracies)

# Evaluate the trained model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

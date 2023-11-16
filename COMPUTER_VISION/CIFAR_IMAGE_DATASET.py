# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and split the CIFAR-10 dataset into training and testing sets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display one image from the dataset
IMG_INDEX = 7  # Change this to view other images
plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

# Create a Convolutional Neural Network (CNN) model
model = models.Sequential()

# Add a Conv2D layer with 32 filters, a 3x3 kernel, and ReLU activation
# Input shape is (32, 32, 3) for RGB images
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# Add a MaxPooling2D layer with a 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))

# Add another Conv2D layer with 64 filters and ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add another MaxPooling2D layer
model.add(layers.MaxPooling2D((2, 2)))

# Add a third Conv2D layer with 64 filters and ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output from the convolutional layers
model.add(layers.Flatten())

# Add a Dense layer with 64 units and ReLU activation
model.add(layers.Dense(64, activation='relu'))

# Add the output layer with 10 units (number of classes in CIFAR-10)
model.add(layers.Dense(10))

# Display a summary of the model architecture
model.summary()

# Compile the model with Adam optimizer, SparseCategoricalCrossentropy loss, and accuracy metric
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model on the training data for 4 epochs
history = model.fit(train_images, train_labels, epochs=4, validation_data=(test_images, test_labels))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Print the test accuracy
print(test_acc)


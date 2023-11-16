# Imports
import os
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras

# Dataset

tfds.disable_progress_bar()

# Split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str  # Creates a function object that we can use to get labels

# Display 2 images from the dataset
for image, label in raw_train.take(5):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

# Data preprocessing
IMG_SIZE = 160  # All images will be resized to 160x160


def format_example(input_name, label):
    """
    Returns an image that is reshaped to IMG_SIZE
    """
    input_name = tf.cast(input_name, tf.float32)
    input_name = (input_name / 127.5) - 1
    input_name = tf.image.resize(input_name, (IMG_SIZE, IMG_SIZE))
    return input_name, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for img, label in raw_train.take(2):
    print("Original shape:", img.shape)

for img, label in train.take(2):
    print("New shape:", img.shape)

# Picking a Pretrained Model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

for image, _ in train_batches.take(1):
    pass

feature_batch = base_model(image)
print(feature_batch.shape)

# FREEZING THE BASE
base_model.trainable = False

# Adding our Classifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.summary()

# Training the Model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# We can evaluate the model right now to see how it does before training it on our new images
initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

# Now we can train it on our images
history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)

# Save the model
model.save("dogs_vs_cats.h5")

# We can save the model and reload it at any time in the future
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

# We can save the model and reload it at any time in the future
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')


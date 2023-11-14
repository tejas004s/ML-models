# Setup and Imports

# Import necessary libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

# Load dataset.

# Training data
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
# Testing data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Feature Columns

# Define categorical and numeric feature columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Create feature columns based on unique values in categorical columns and numeric columns
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

# Input Function

# Define a function to create input functions for training and evaluation
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        # Create a TensorFlow Dataset object from data and labels
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)  # Randomize order of data if shuffle is True
        ds = ds.batch(batch_size).repeat(num_epochs)  # Split dataset into batches and repeat for epochs
        return ds
    return input_function

# Create input functions for training and evaluation
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Creating the Model

# Create a linear estimator model using the feature columns
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Training the Model
linear_est.train(train_input_fn)  # Train the model using the training input function
result = linear_est.evaluate(eval_input_fn)  # Evaluate the model using the evaluation input function

clear_output()  # Clear console output
print(result['accuracy'])  # Print the accuracy of the model

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')


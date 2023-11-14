# Imports and Setup
import tensorflow_probability as tfp
import tensorflow as tf

# Distribution Variables
tfd = tfp.distributions  # Making a shortcut for later use

# Define the initial distribution of states
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])

# Define the transition distribution between states
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])

# Define the observation distribution for each state
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# Model
# Create a Hidden Markov Model with the specified distributions
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)  # Number of time steps in the model

# Temperature
# Calculate the mean of the model (output the expected values of the states over time)
mean = model.mean()

# Due to TensorFlow's computation graph, we need to use a session to evaluate the tensor
# In the new version of TensorFlow, use tf.compat.v1.Session() instead of tf.Session()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())


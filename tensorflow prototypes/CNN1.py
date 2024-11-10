#=========================================================================
# Title:            Convolutional Neural Network 1 Tensorflow Prototype
# Author:           Thomas M. Marshall
# Student Number:   University of Pretoria 18007563
# Last Updated:     20 November 2021
#=========================================================================

# Libraries
import tensorflow as tf

class_names = ["AMDSB", "AMLSB", "AMUSB", "WBFM", "2FSK", "GFSK", "GMSK", "BPSK", "QPSK", "PSK_8", "QAM_8", "QAM_16", "QAM_32"]
batch_size = 1
epochs = 50
dataset = "MULTIFREQ_small_S.csv"

# Helper functions
def stackFeatureVector(features, labels):
  # This gives out a list of batch_size dimensional tensors.
  batch = list(features.values())

  # Now stack them on top of eachother
  features = tf.stack(batch, axis=1)

  # Now reshape the tensor 
  features = tf.reshape(features, (batch_size, 2, 128))

  # Expand dims to appease tensorflow gods
  features = tf.expand_dims(features, axis=-1)

  return features, labels

# Make TF dataset out of CSV dataset
train_dataset = tf.data.experimental.make_csv_dataset(dataset, batch_size, label_name="modulation_id", header=True, num_epochs=1).map(stackFeatureVector)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, activation='relu', kernel_size=(1,3), strides=(1,1), input_shape=(2, 128, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(13),
])

model.summary()

model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_dataset, epochs=epochs)

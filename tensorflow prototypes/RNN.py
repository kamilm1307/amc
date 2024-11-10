#=========================================================================
# Title:             Tensorflow Prototype
# Author:           Thomas M. Marshall
# Student Number:   University of Pretoria 18007563
# Last Updated:     20 November 2021
#=========================================================================

# Libraries
import tensorflow as tf

class_names = ["AMDSB", "AMLSB", "AMUSB", "WBFM", "GFSK", "GMSK", "BPSK", "QPSK", "PSK_8", "QAM_8", "QAM_16", "QAM_32"]
batch_size = 1
epochs = 50
dataset = "MULTIFREQ_small_S.csv"

# Helper functions
def stackFeatureVector(features, labels):
  features = tf.expand_dims(tf.stack((list(features.values())), axis=1), axis=-1)
  return features, labels

# Make TF dataset out of CSV dataset
train_dataset = tf.data.experimental.make_csv_dataset(dataset, batch_size, label_name="modulation_id", header=True, num_epochs=1).map(stackFeatureVector)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(512, input_shape=(256,1)),
    tf.keras.layers.Dense(1024*2, activation='relu'),
    tf.keras.layers.Dense(12),
])

model.summary()
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_dataset, epochs=epochs)
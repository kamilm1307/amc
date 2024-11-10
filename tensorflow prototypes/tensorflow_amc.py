#=====================================================================================
# Title:        Automatic Modulation Identification System
# Author:       Thomas M. Marshall
# Last Updated: 20 November 2021
# Usage:        Command line parameters: center frequency, modelname
#=====================================================================================

# Libraries
import os
import sys
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress Tensorflow 
import tensorflow as tf
from tensorflow import keras
import rtlsdr as rtl

modulations = ["AMDSB ", "AMLSB ", "AMUSB ", "WBFM ", "2FSK ", "GFSK ", "GMSK ","2PSK ", "4PSK ", "8PSK ", "8QAM ", "16QAM ", "32QAM "]

def init_SDR(rx_freq):
    # RTL-SDR Object
    SDR = rtl.RtlSdr()			# RTL-SDR object instantiation
    SDR.sample_rate = 2.4e6		# Sampling rate to maximum of RTL-SDR
    SDR.center_freq = rx_freq	# Center frequency to sample at on RTL-SDR
    SDR.gain = 0.0				# Receiver gain of RTL-SDR in dB
    return SDR

def init_Model(model_name):
    model = tf.keras.models.load_model('./trained-models/' + model_name + '.h5', compile=False)
    return model

def AMC(rx_freq, mode, model_name):
    # Initialize radio and classifier
    RTL_SDR = init_SDR(rx_freq)
    amc_proto = init_Model(model_name)

    print("Automatic Modulation Classification System: Tensorflow Prototype")
    print("Scanning at", rx_freq, "Hz")
    
    debug_counter = 0
    while(1):
        debug_counter += 1
        # Perform ensemble approach for N=100
        instances = np.zeros(len(modulations))
        for n in range(80):
            # Read samples from RTL-SDR
            samples = RTL_SDR.read_samples(256)[:128]

            # Convert samples to array of I and Q
            sample_set = []
            for s in samples:
                sample_set.append((s.real))
                sample_set.append((s.imag))

 
            # CNN tensor reshaped input
            inputs = (tf.convert_to_tensor(sample_set))
            inputs = tf.reshape(inputs, [2, 128])
            inputs = tf.expand_dims(inputs, axis=-1)
            inputs = tf.expand_dims(inputs, axis=0)
            
            # Calculate the output
            output = np.array(amc_proto(inputs))[0]

            # Process the output
            class_ = (np.where(output == max(output))[0][0])

            # Update list
            instances[class_] += 1

        print("Modulation classified:", modulations[(np.where(instances == max(instances))[0][0])], '\n', instances, "with", round(max(instances)/sum(instances),2),'\t\t' , end='\r')

# Main function
if __name__ == '__main__':
    AMC(rx_freq = int(float(sys.argv[1])), model_name = sys.argv[3])

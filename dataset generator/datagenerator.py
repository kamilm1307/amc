#=========================================================================
# Title:            SDR Data Generator
# Author:           Thomas M. Marshall
# Student Number:   University of Pretoria 18007563
# Last Updated:     20 November 2021
#=========================================================================

# Libraries
import csv
import time
import subprocess
import numpy as np
import rtlsdr as rtl
import random as random
import matplotlib.pyplot as plt
from matplotlib import rc

# Enable LateX and set font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Operational parameters
DEBUG_MODE = False
CONTINUATION = False
SET_SIZE = 250
dataset_name = 'dataset.csv'

def dataGenerator():
    print("--------------------------")
    print("AMC Project Data Generator")
    print("--------------------------")

    start_global = time.perf_counter_ns()	# Start timer

    # Initialization    
    N = 128	    # Number of IQ samples taken
    M = 1       # Number of times N is read for each modulator

    print("This instance will generate and record", str(M*SET_SIZE), "unique samples")
    print("Estimated Training Set Generation Time:", round(((SET_SIZE*1.45)), 2), "seconds or", round(((SET_SIZE*1.45))/60, 2), "minutes", "or", round((((SET_SIZE*1.45))/60)/60, 2), "hours")

    ms_delay = 1000		# Delay in ms to allow for transmitter steady state
    tx_freq = 30e6 		# Transmitter frequency in Hz - Will remain constant througout

    # RTL-SDR Object
    SDR = rtl.RtlSdr()			# RTL-SDR object instantiation
    SDR.sample_rate = 2.4e6		# Sampling rate to maximum of RTL-SDR
    SDR.center_freq = tx_freq	# Center frequency to sample at on RTL-SDR
    SDR.gain = 0.0				# Receiver gain of RTL-SDR in dB

    # List of 12 valid modulator scripts
    modulators = ["AMDSB", "AMLSB", "AMUSB", "WBFM", "GFSK", "GMSK", "PSK_2", "PSK_4", "PSK_8", "QAM_8", "QAM_16", "QAM_32", "FSK_2"] 
    modulations =["AMDSB", "AMLSB", "AMUSB", "WBFM", "GFSK", "GMSK", "2PSK", "4PSK", "8PSK", "8QAM", "16QAM", "32QAM", "2FSK"] 
    modulation_distribution = np.zeros(len(modulators))

    # Open dataset file, and append or start over depending on the continuation variable            
    if(CONTINUATION == 1):
        dataset = open(dataset_name, mode='a') # Change mode between w and a
    else:
        dataset = open(dataset_name, mode='w') # Change mode between w and a

    # Create file writer
    fileWriter = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write headers if the file is started over
    if(CONTINUATION == 0):
        # Write headers dynamically based on size of input
        headers = ['modulation_id']
        for n in range(N*2):
            headers.append("f_" + str(n))
        fileWriter.writerow(headers)
        print("\nStarting new dataset")
    else:
        print("Adding to existing dataset")

    # Transmit and receive K number of signals from the HACKRF to the RTL-SDR
    for num in range(SET_SIZE):
        start_t = time.perf_counter_ns()	# Start timer

        # Choose random uniformly distributed modulator
        random_modulator = random.choice(modulators)

        # Count up that this modulator was used
        modulation_distribution[modulators.index(random_modulator)] += M

        # Choose random parameters
        param_3 = ''
        tx_gain = 0
        if(random_modulator in ["AMDSB", "AMLSB", "AMUSB", "WBFM"]):
            # Analog Modulations: Need to choose audio file
            param_3 = random.choice(["castleonthehill","ludovico","psychosocial","radiovoice","requiem"])
        elif(random_modulator in ["FSK_2"]):
            # Frequency shift: choose oscillator sensitivity in radians/volts/sec
            param_3 = str(random.choice(np.arange(5e6, 30e6, 5e6)))
        else:
            # Digital Modulations: Need to choose excess bandwidth
            param_3 = str(random.choice(np.arange(0.1, 1.0, 0.1)))

        tx_gain = random.choice(np.arange(0, 5, 1))
        tx_freq = random.choice([90e6, 250e6, 500e6, 750e6, 1000e6, 1250e6, 1500e6, 1700e6])

        # Tune RTL-SDR to chosen carrier frequency
        SDR.center_freq = tx_freq

        print("Set", '{:^8}'.format(str(num)), ":", '{:^8}'.format(random_modulator), '{:^8}'.format(str(tx_freq//10**6)+"MHz"), ' -> ', '{:^4}'.format(str(round(((num+1)/SET_SIZE)*100, 2)) + '%'), end = '\r')

        # Open the process and broadcast random modulator from the HACKRF, and supress outputs from all subprocesses created
        modulation = subprocess.Popen(["python3", "modulators/" + random_modulator + ".py", str(tx_gain), str(tx_freq), str(param_3)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Delay - wait for transmitter steady state
        time.sleep(ms_delay * 10**-3)

        # Induce additional random delay for variability in audio with analog modulations
        if(random_modulator in ["AMDSB", "AMLSB", "AMUSB", "WBFM", "FSK_2"]):
            time.sleep(random.choice(np.arange(500, 1500, 50)) * 10**-3)
        
        # Collect M sets of N samples from the RTL-SDR
        for m in range(M):      
            # Record with the RTL-SDR
            warmup = SDR.read_samples(1024)
            samples = SDR.read_samples(1024)[1024-128:]
            
            # Start with the class in record writing
            sample_set = [modulators.index(random_modulator)]
            for s in samples:
                sample_set.append(s.real)
                sample_set.append(s.imag)
            fileWriter.writerow(sample_set)

        # Terminate the transmission process
        modulation.kill()

    # Close RTL-SDR when program terminates
    SDR.close()

    # Close the file
    dataset.close()

    # Total execution time
    measured_t = (time.perf_counter_ns() - start_global)/10**9
    print("Measured Training Set Generation Time:", round((measured_t), 2), "seconds or", round((measured_t)/60, 2), "minutes", "or", round(((measured_t)/60)/60, 2), "hours")
    print("\nThe modulation distribution is:\n", modulators, '\n',modulation_distribution, '\n')
    
    # Plot figure of class distribution of the generated signals
    fig, ax = plt.subplots(figsize = (8,6))
    idx = np.asarray([i for i in range(len(modulations))])
    ax.bar(idx, modulation_distribution)
    ax.set_xticks(idx)
    ax.set_xticklabels(modulations, rotation=65)
    ax.set_xlabel(r'Modulators')
    ax.set_ylabel(r'Number of samples')
    ax.set_title(r'Dataset distribution')
    fig.tight_layout()
    plt.savefig("./figures/dist.png", dpi=300)

# Main function
if __name__ == '__main__':
    dataGenerator()

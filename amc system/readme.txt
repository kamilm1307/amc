#=========================================================================
# Title:            Automatic Modulation Identification System Readme
# Author:           Thomas M. Marshall
# Student Number:   University of Pretoria 18007563
# Last Updated:     20 November 2021
#=========================================================================

Description:    This software automatically identifies modulation schemes 
                on a signal channel with no information other than the 
                signal itself. 

                The model can identify thirteen different RF modulation 
                schemes. These categories are: AMDSB, AMLSB, AMUSB,
                WBFM, 2FSK, GFSK, GMSK, 2PSK, 4PSK, 8PSK, 8QAM, 
                16QAM and 32QAM.

                The model works best when used at a signal to noise ratio
                of 20 dB (as is indicated by the software), and at a lower
                frequency relative to the entire range of the RTL-SDR.

Usage:          To operate the system the following command may be used:
                python3 amc.py modelname fc gain bw N
                    modelname   - Trained model to load (e.g. amcnet)
                    fc          - Carrier frequency (Hz)
                    gain        - Gain (dB)
                    bw          - Bandwidth (Hz)
                    N           - Number of measurements taken
                
                For example, the system can be started with this command:
                	python3 amc.py amcnet 90e6 0 2.4e6 4
                	
                To exit the system, enter CTRL + C into the terminal.

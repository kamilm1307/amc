# Automatic Modulation Classification using CNNs and SDR

## Overview
This project implements a real-time automatic modulation classification (AMC) system using Convolutional Neural Networks (CNNs) and Software Defined Radio (SDR). The system can identify various digital and analog modulation schemes from live radio signals captured using an RTL-SDR device. 

If you are a mega-outlier of some sort, and happen to have the trifecta of an interest in ML, an SDR and an Nvidia GPU; weclome :)

At the time (2021), implementing Conv nets from scratch with some GPU acceleration, and applying it to a novel domain such as SDR signal processing was a sizable challenge. The keen observer will notice that `amcnet.py` was updated in 2024. This is because at the time of doing this project, I was running out of time, but suspected there was a syntatical error causing a logical error in my deep learning. I didn't have time to hunt this down until now. This has periodically been keeping me awake at night for four years. 

### The Old Problems
1. Adam Optimizer Implementation: The original code failed to properly initialize the Adam optimizer's beta parameters (beta1 and beta2) as instance variables.
2. Missing Learning Rate: A critical hyperparameter (learning rate) was undefined in the original implementation, causing potential training failures.
3. Memory Inefficiency: The original im2col implementation created unnecessary array copies, leading to potential memory issues with large inputs.
4. Inconsistent Dropout: The dropout implementation had inconsistent scaling between forward and backward passes.
5. Error Handling: The original code lacked robust error handling for file operations and SDR interactions.
6. History Tracking: The loss history tracking was referenced but never initialized, causing potential crashes during model saving.

Publishing this repository correcting the mistakes of my 20 year old self is cathartic, and demonstrates to myself how much I have progressed since. 

## Features
- Real-time modulation classification using custom CNN architecture
- Support for 13 different modulation types:
  - Analog Modulations: AMDSB, AMLSB, AMUSB, WBFM
  - Digital Modulations: GFSK, GMSK, 2PSK, 4PSK, 8PSK
  - Digital Amplitude Modulations: 8QAM, 16QAM, 32QAM
  - Frequency Shift Keying: 2FSK
- GPU acceleration support for faster processing
- Custom dataset generation capabilities
- Real-time signal processing and classification
- Signal-to-Noise Ratio (SNR) estimation

## Requirements
- Python 3.x
- RTL-SDR device
- Required Python packages:
  ```
  numpy
  numexpr
  rtlsdr
  termcolor
  pycuda (optional, for GPU acceleration)
  scikit-cuda (optional, for GPU acceleration)
  ```

## Usage

1. Run the AMC system:
```bash
python amc.py <model_name> <center_frequency> <gain> <bandwidth> <num_measurements>
```

Example:
```bash
python amc.py amcnet 90e6 20.0 2.4e6 5
```

Parameters:
- `model_name`: Name of the pre-trained model file (*.npy)
- `center_frequency`: Center frequency in Hz (e.g., 90e6 for 90 MHz)
- `gain`: SDR gain in dB (typically 0-40)
- `bandwidth`: Sampling bandwidth in Hz
- `num_measurements`: Number of measurements to average for classification

2. Generate training dataset:
```bash
python datagenerator.py
```

## Implementation Details

### CNN Architecture
- Input Layer: Complex I/Q samples (2x128)
- Two Convolutional Layers:
  - Conv1: 256 filters (1x3)
  - Conv2: 80 filters (2x3)
- Two Dense Layers:
  - Dense1: 128 neurons
  - Dense2: 13 neurons (output layer)
- Activation Functions: ReLU (hidden layers), Softmax (output)
- Dropout rate: 0.1

### Signal Processing
- Real-time I/Q sample acquisition via RTL-SDR
- Automatic gain control and SNR estimation
- Signal preprocessing and normalization
- GPU acceleration for matrix operations (optional)

## Performance
- Real-time classification capability
- Classification accuracy varies with SNR
- Processing time: typically <100ms per classification
- GPU acceleration can provide significant speedup for matrix operations

## License
See [LICENSE](LICENSE) file for details.

## Author
Thomas M. Marshall  

## Citation
If you use this work in your research, please cite:
```
Marshall, T.M. (2021). Automatic Modulation Identification System Using Convolutional Neural Networks and Software Defined Radio. University of Pretoria, Final Year Project.
```

## Disclaimer
The audio samples used for amplitude and frequency modulation testing are utilized under fair use for academic research purposes only. All audio content remains the property of their respective copyright holders. This project makes no claim of ownership and is not intended for commercial use.
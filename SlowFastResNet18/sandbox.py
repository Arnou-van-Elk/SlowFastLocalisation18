import pickle
import os
import numpy as np


# Define the directory containing the NPZ files
directory = r'C:\Users\arnou\Documents\Radboud\Thesis\slowfast18\LogMelSpectrograms\LogMelSpectrograms'

# List all the files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.npz')]

# Check if there are any npz files in the directory
if files:
    # Process each npz file in the directory
    for file in files:
        file_path = os.path.join(directory, file)

        # Load the content of the npz file using numpy
        data = np.load(file_path)
        
        # Assuming there is only one item in each npz file
        key = list(data.keys())[0]  # Get the key for the stored array

        # Retrieve the log Mel spectrogram
        log_mel_spectrogram = data[key]

        # Optionally, print the key and the shape of the log Mel spectrogram
        print(f"File: {file}")
        print(f"Key: {key}")
        print(f"Shape of the Log Mel Spectrogram: {log_mel_spectrogram.shape}")
        print(f"Data type: {log_mel_spectrogram.dtype}")

        # Close the loaded npz file
        data.close()
else:
    print("No NPZ files found in the directory.")
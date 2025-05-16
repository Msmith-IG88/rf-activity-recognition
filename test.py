import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from PIL import Image, ImageOps
import csv

# === CONFIG ===
data_dir = "../data/Real_data"
output_dir = "../data/real_spectrograms"
label_csv_path = "../data/real_labels.csv"
os.makedirs(output_dir, exist_ok=True)

window_size = 100  # 2 seconds at 50 Hz
step_size = 50     # 50% overlap
fs = 50            # Sampling rate
channel_indices = [0, 10, 20]  # ACC:X, GYR:Y, GYR:Z (example)

X_windows = []
y_labels = []

# === Step 1: Collect all subject files ===
log_files = glob.glob("../data/Real_data/subject*.log")
print(f"Found {len(log_files)} log files.")

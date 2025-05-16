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

# === Step 2: Extract windows from all files ===
for log_file in log_files:
    try:
        df = pd.read_csv(log_file, sep='\t', header=None)
        features = df.iloc[:, 2:-1].astype(float)  # Columns 2 to 118
        labels = df.iloc[:, -1].astype(int)        # Last column

        for start in range(0, len(features) - window_size + 1, step_size):
            end = start + window_size
            window = features.iloc[start:end].values
            label_window = labels[start:end]

            if len(label_window) == 0:
                continue

            majority_label = label_window.mode()[0]
            if majority_label == 0:
                continue

            X_windows.append(window)
            y_labels.append(majority_label)

    except Exception as e:
        print(f"Error processing {log_file}: {e}")

X_windows = np.array(X_windows)
y_labels = np.array(y_labels)
print(f"✅ Extracted {len(X_windows)} labeled windows.")

# === Step 3: Generate and save spectrograms ===
csv_rows = []

for i, window in enumerate(X_windows):
    channels = []
    for ch in channel_indices:
        f, t, Sxx = spectrogram(window[:, ch], fs=fs, nperseg=32, noverlap=16)
        Sxx = np.log(Sxx + 1e-10)

        if np.any(np.isnan(Sxx)) or Sxx.max() == Sxx.min():
            continue  # Skip broken spectrograms

        norm_Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min()) * 255
        channels.append(norm_Sxx.astype(np.uint8))

    if len(channels) == 3:
        img_shape = np.min([c.shape for c in channels], axis=0)
        stacked = np.stack([c[:img_shape[0], :img_shape[1]] for c in channels], axis=-1)
        img = Image.fromarray(stacked)
        img = ImageOps.fit(img, (128, 128), Image.Resampling.BICUBIC)

        filename = f"real_{i}.png"
        img.save(os.path.join(output_dir, filename))
        csv_rows.append([filename, y_labels[i]])

# === Step 4: Save label file ===
with open(label_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(csv_rows)

print(f"✅ Saved {len(csv_rows)} spectrograms to {output_dir}")
print(f"✅ Labels written to {label_csv_path}")

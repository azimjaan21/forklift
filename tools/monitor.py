import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import os

csv_path = r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\forklift\runs\detect\train\results.csv'

plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

while True:
    if os.path.exists(csv_path):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()  # Clean column names
            print("Columns in CSV:", df.columns.tolist())  # Debug column names

            # Check for required columns
            required_columns = ['epoch', 'train/box_loss', 'val/box_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
            if not all(col in df.columns for col in required_columns):
                print("Required columns not found. Retrying...")
                sleep(1)
                continue

            # Extract metrics
            epochs = df['epoch']
            train_loss = df['train/box_loss']
            val_loss = df['val/box_loss']
            mAP50 = df['metrics/mAP50(B)']
            mAP5095 = df['metrics/mAP50-95(B)']

            # Clear previous plots
            for a in ax.ravel():
                a.clear()

            # Plot Training vs Validation Box Loss
            ax[0, 0].plot(epochs, train_loss, label='Train Box Loss', color='blue')
            ax[0, 0].plot(epochs, val_loss, label='Val Box Loss', color='orange')
            ax[0, 0].set_title('Box Loss')
            ax[0, 0].set_xlabel('Epoch')
            ax[0, 0].set_ylabel('Loss')
            ax[0, 0].legend()

            # Plot mAP50
            ax[0, 1].plot(epochs, mAP50, label='mAP@50', color='green')
            ax[0, 1].set_title('mAP@50')
            ax[0, 1].set_xlabel('Epoch')
            ax[0, 1].set_ylabel('mAP')
            ax[0, 1].legend()

            # Plot mAP50-95
            ax[1, 0].plot(epochs, mAP5095, label='mAP@50-95', color='purple')
            ax[1, 0].set_title('mAP@50-95')
            ax[1, 0].set_xlabel('Epoch')
            ax[1, 0].set_ylabel('mAP')
            ax[1, 0].legend()

            plt.tight_layout()
            plt.pause(1)  # Update the plot every second

        except Exception as e:
            print(f"Error while processing file: {e}")
            sleep(1)
            continue
    else:
        print("File not found. Retrying...")

    sleep(1)  # Wait for 1 second before checking again

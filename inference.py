import os
import cv2
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from train import load_or_train_model, MODEL_SAVE_PATH, get_class_mapping, log


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on images in the specified directory")
    parser.add_argument('--input', type=str, required=True, help="Path to the directory containing images")
    return parser.parse_args()


def predict():
    args = parse_arguments()
    directory = args.input
    model = load_or_train_model(MODEL_SAVE_PATH)
    class_mapping = get_class_mapping()

    if not os.path.exists(directory) or not os.path.isdir(directory):
        print(f"The file {directory} does not exist or is not a directory")

    for filename in os.listdir(directory):
        if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith("png"):
            try:
                img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)
                img = cv2.bitwise_not(img)
                # img = np.invert(np.array([img]))
                img = img.reshape(1, 30, 30, 1)
                img = img.astype('float32') / 255.0

                prediction = model.predict(img, verbose=0)

                predicted_class_idx = np.argmax(prediction)
                predicted_class = class_mapping[predicted_class_idx]
                predicted_ascii_code = ord(predicted_class)

                print(f'{predicted_ascii_code:03}, {os.path.join(directory, filename)}')

            except RuntimeError:
                print(f"Error processing {filename}")


predict()

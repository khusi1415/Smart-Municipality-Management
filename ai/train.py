"""Train the image classifier.
Folder layout:  dataset/garbage/*.jpg  dataset/pothole/*.jpg  dataset/street_light/*.jpg  dataset/water_leakage/*.jpg
Run:  python ai/train.py            (from the project folder)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai import image_model

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    print(image_model.train(folder))
    print("Model saved to", image_model.MODEL_PATH)

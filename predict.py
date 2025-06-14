import os 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

# Load model
model = load_model('model/leaf_disease_model.h5')

# Load class names
class_names = sorted(list(os.listdir('dataset')))

# Load image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    print(f"Predicted Class: {class_names[index]}")

# Run prediction
if __name__ == "__main__":
    img_path = sys.argv[1]  # e.g., python predict.py test.jpg
    predict_image(img_path)

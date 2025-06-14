import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uuid

app = Flask(__name__)
model = load_model('model/leaf_disease_model.h5')
class_names = sorted(os.listdir('dataset'))

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    # Create a unique filename for the graph
    graph_filename = f"static/graph/{uuid.uuid4().hex}.png"
    
    # Generate the bar chart
    plt.figure(figsize=(10, 4))
    plt.bar(class_names, prediction, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Confidence")
    plt.title("Prediction Confidence for Each Class")
    plt.tight_layout()
    plt.savefig(graph_filename)
    plt.close()

    return predicted_class, graph_filename

@app.route('/')
def index():
    sample_images = []
    test_images_path = 'static/test_images'
    if os.path.exists(test_images_path):
        sample_images = os.listdir(test_images_path)
    return render_template('index.html', sample_images=sample_images)

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected.", 400

    if file:
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        prediction, graph_path = predict_disease(file_path)
        return render_template('result.html', prediction=prediction, image_path=file_path, graph_path=graph_path)

if __name__ == '__main__':
    os.makedirs('static/graph', exist_ok=True)
    app.run(debug=True)

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load trained model
model = load_model("model.h5")

# Load class names from train folder
class_names = sorted(os.listdir("train"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']
        if file.filename == '':
            return "No image selected"

        # Clean filename
        filename = secure_filename(file.filename.replace(" ", "_"))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file
        file.save(filepath)
        print("✅ File saved to:", filepath)

        # Confirm file saved
        if not os.path.exists(filepath):
            return "❌ ERROR: File not found after saving: " + filepath

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        # Fix file path for browser
        image_url = url_for('static', filename=('uploads/' + filename).replace("\\", "/"))

        return render_template('index.html', prediction=predicted_class, image_path=image_url)

    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
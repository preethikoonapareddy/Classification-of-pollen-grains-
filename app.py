import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = load_model("model/final_pollen_model.keras", compile=False)

# Load class labels
label_file = "class_labels.txt"
with open(label_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Flask app setup
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/logout.html')
def logout():
    return render_template('logout.html')

@app.route("/test")
def test_background():
    return render_template("test.html")

@app.route('/result', methods=["GET", "POST"])
def result():
    if request.method == 'POST':
        f = request.files['image']

        # Create upload folder if it doesn't exist
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)

        # Preprocess image
        try:
            img = load_img(filepath, target_size=(128, 128))
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)

            pred = model.predict(x)
            pred_index = np.argmax(pred)

            # Validate index
            if pred_index < len(class_names):
                result = class_names[pred_index]
            else:
                result = "Prediction index out of bounds."

        except Exception as e:
            result = f"Error processing image: {str(e)}"

        return render_template('prediction.html', pred=result)

if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)


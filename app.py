from flask import Flask, request
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

MODEL = tf.keras.models.load_model("../final123/models/9/1")

CLASS_NAMES = ["Anthracnose", "Bacterial Canker", "Black Soothy Mold", "Cutting Weevil", "Die Back", "Gail Midge", "Healthy", "Powdery Mildew", "Sooty Mould" ]
#["Anthracnose", "Bacterial Canker", "Gail Midge", "Healthy", "Powdery Mildew", "Sooty Mould" ]
RECOMMENDATIONS = None

@app.route("/ping", methods=['GET'])
def ping():
    return "Hello, I am alive", 200

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['file']
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]

    confidence = np.max(predictions[0])
    conf = int(float(confidence) * 100)
	
    if predicted_class == "Cutting Weevil" or predicted_class == "Die Back" or predicted_class == "Black Soothy Mold":
        return {
            "unable": True,
            "class": predicted_class,
            'confidence': float(confidence)
        }
    elif conf < 90:
        return {
            "unable": True,
            "class": predicted_class,
            'confidence': float(confidence)
        }
    else:
        return {
        'class': predicted_class,
        'confidence': float(confidence)
        }


if __name__ == "__main__":
	app.run()

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class_names = ["Anthracnose", "Bacterial Canker", "Black Soothy Mold", "Cutting Weevil", "Die Back", "Gail Midge", "Healthy", "Powdery Mildew", "Sooty Mould" ]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image = read_file_as_image(await file.read())
    print(image)

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    )

    print("before scaling:", image)
    image = image/255 # normalize the image in 0 to 1 range
    print("after scaling:", image)

    predicted_class, confidence = predict_using_regular_model(image)
    return {"class": predicted_class, "confidence": confidence}

def predict_using_regular_model(img):
    model = tf.keras.models.load_model("../mango3.h5")
    img_array = tf.expand_dims(img, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


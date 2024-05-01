from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import joblib

import numpy as np

from sklearn.ensemble import RandomForestClassifier
import uuid
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50V2
# Now you can use Pillow functions, such as load_img


# load the models when import "predictions.py"
model_elbow_frac = tf.keras.models.load_model(r"C:/Users/venky/PycharmProjects/ML model/weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model(r"C:/Users/venky/PycharmProjects/ML model/weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model(r"C:/Users/venky/PycharmProjects/ML model/weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model(r"C:/Users/venky/PycharmProjects/ML model/weights/ResNet50_BodyParts.h5")

# categories for each result by indexC:\Users\venky\PycharmProjects\ML model\weights\ResNet50_BodyParts.h5

#   0-Elbow     1-Hand      2-ShoulderC:\Users\venky\PycharmProjects\ML model\weights\ResNet50_Shoulder_frac.h5
categories_parts = ["Elbow", "PALM", "Shoulder"]

#   0-fractured     1-normal
categories_fracture = ['fractured', 'normal']


def predict(img, model="Parts"):
    # handle invalid model names
    if model not in ["Parts", "Elbow", "Hand", "Shoulder"]:
        raise ValueError(f"Invalid model name: {model}")

    # Make prediction using the model
    size = (224, 224)  # Target size
    temp_img = Image.open(img)
    temp_img = temp_img.resize(size)

    # Convert image to NumPy array and normalize
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # load the appropriate model based on the input
    if model == 'Parts':
        chosen_model = model_parts
    elif model == 'Elbow':
        chosen_model = model_elbow_frac
    elif model == 'Hand':
        chosen_model = model_hand_frac
    elif model == 'Shoulder':
        chosen_model = model_shoulder_frac

    prediction = np.argmax(chosen_model.predict(x), axis=1)
    # choose the category and get the string prediction
    if model != "Parts":
        prediction_str = categories_fracture[prediction.item()]
    else:
        prediction_str = categories_parts[prediction.item()]
    print(prediction_str)
    return prediction_str
app = Flask(__name__)

# Specify the directory where uploaded datasets will be saved
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model_path = 'xray_model_random_forest.pkl'
model = joblib.load(model_path)

# Data augmentation for the uploaded image
datagen = image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Extract features from the pre-trained ResNet50V2 model
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def analyze_dataset(file_path):
    # Load and preprocess the image
    img_array = preprocess_image(file_path)

    # Apply data augmentation
    augmented_images = []
    for batch in datagen.flow(img_array, batch_size=1):
        augmented_images.append(batch)
        if len(augmented_images) >= 5:  # Generate 5 augmented images
            break

    predictions = []
    for img in augmented_images:
        features = base_model.predict(img)
        features = np.reshape(features, (1, 7 * 7 * 2048))
        prediction = model.predict(features)
        predictions.append(prediction[0])

    avg_prediction = np.mean(predictions)

    if avg_prediction >= 0.5:
        result = ['The image is affected by pneumonia.']
    else:
        result = ['The image is not affected by pneumonia.']

    return result


@app.route('/')
def home():

    return render_template('home.html')
@app.route('/main')
def main():
    return render_template('main.html')
@app.route('/bone')
def bone():
    return render_template('bone.html')
@app.route("/index", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return 'No selected file'

        if file:
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
            file_path = "static/" + unique_filename
            file.save(file_path)
            # Analyze the uploaded image
            result = analyze_dataset(file_path)
            print(file_path)
        return render_template('report.html', result=result, image_file=file_path)
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def make_prediction():
    if request.method == 'POST':
        file = request.files['file']
        # Generate a unique filename using uuid
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
        filename = "static/" + unique_filename
        print(filename, file)

        try:
            # Save the file
            file.save(filename)
            model = request.form['model']
            print(model)
            prediction = predict(filename, model)
            print(prediction)
            return render_template('bone.html', prediction=prediction, image_file=filename)
        except Exception as e:
            print(e)
            return render_template('bone.html', error="Error processing image. Please upload a valid image file.",
                                   image_file=None)

app.run(debug=True)
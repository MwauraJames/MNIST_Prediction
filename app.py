import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("my_mnist_model.keras")

st.title("Take a Photo")

photo = st.camera_input("Take a picture")

def prepare_my_image(image):
    # 1. Load, grayscale, and resize
        # 1. Convert to grayscale
    image = image.convert("L")

    # 2. Resize to 28x28
    image = image.resize((28, 28))

    # 3. Convert to numpy array
    img_array = np.array(image)
    # 2. Invert the colors (White paper -> Black, Dark ink -> White)
    img_array = 255.0 - img_array

    # 3. Normalize to decimals (0.0 to 1.0)
    img_array = img_array / 255.0

    # --- 4. THE NEW FIX: THRESHOLDING ---
    # Any annoying gray background noise (below 0.5) gets crushed to pure black (0.0)
    img_array = np.where(img_array < 0.5, 0.0, img_array)

    # Any light gray ink gets boosted to pure bright white (1.0)
    img_array = np.where(img_array > 0.5, 1.0, img_array)

    # --- 5. VISUAL CHECK ---
    # Let's plot it right inside the function so you can immediately see if it worked


    # 6. Add the Batch dimension for the model
    final_input = tf.expand_dims(img_array, axis=0)

    return final_input


if photo is not None:
    image = Image.open(photo)
    
    my_ready_image = prepare_my_image(image)
    # Convert to numpy
if st.button("Predict"):
    raw_predictions = model.predict(my_ready_image)
    predicted_digit = np.argmax(raw_predictions)

    confidence_decimal = np.max(raw_predictions)
    confidence_percentage = confidence_decimal * 100

    st.text(f"The model is {confidence_percentage:.2f}% sure this is a {predicted_digit}!")
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import tensorflow as tf

# Configure the page layout to be wider
st.set_page_config(layout="wide", page_title="Digit Recognizer")

st.title("🔢 Handwritten Digit Recognizer")
st.markdown("Take a photo of a single digit, crop it tightly, and watch the AI guess it!")

# --- 1. CACHING THE MODEL ---
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="mnist_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- 2. THE PREPROCESSING FUNCTION ---
def prepare_my_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image)
    
    img_array = 255.0 - img_array
    img_array = img_array / 255.0

    img_array = np.where(img_array < 0.5, 0.0, img_array)
    img_array = np.where(img_array > 0.5, 1.0, img_array)

    final_input = np.reshape(img_array, (1, 28, 28, 1))
    return final_input

# --- 3. THE UI LAYOUT ---
# Create two main columns: Left for input, Right for results
col1, col2 = st.columns([1, 1])

with col1:
    photo = st.camera_input("Take a picture")

    if photo is not None:
        image = Image.open(photo)
        st.write("Crop the image so the number fills the red box:")
        
        # The Cropper
        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color="red",
            aspect_ratio=(1, 1) 
        )
        
        # Save the cropped image to session state so we don't lose it
        if st.button("Confirm Crop"):
            st.session_state.ready_image = prepare_my_image(cropped_img)

with col2:
    # Only show this section if the user has successfully cropped an image
    if "ready_image" in st.session_state:
        st.subheader("Prediction Area")
        
        if st.button("Run AI Prediction", use_container_width=True, type="primary"):
            # Use a spinner to give visual feedback
            with st.spinner("Analyzing handwriting..."):
                input_data = st.session_state.ready_image.astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                raw_predictions = interpreter.get_tensor(output_details[0]['index'])

                predicted_digit = np.argmax(raw_predictions)
                confidence_percentage = np.max(raw_predictions) * 100

            # Display the result beautifully!
            st.success(f"## I am {confidence_percentage:.1f}% sure this is a {predicted_digit}!")
            
            # Optional: Show the raw probabilities as a bar chart!
            st.bar_chart(raw_predictions[0])
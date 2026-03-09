import tensorflow as tf

# 1. Load your locally trained Keras 3 model
model = tf.keras.models.load_model("my_mnist_model.v2.keras")

# 2. Convert it to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save the new, tiny file
with open("mnist_model.v2.tflite", "wb") as f:
    f.write(tflite_model)
    
print("Successfully created mnist_model.tflite!")
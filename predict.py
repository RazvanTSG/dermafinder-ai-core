import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sys

def predict_single_image(img_path, model_path="models/derma_model_final.h5"):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    classes = ['BCC', 'BKL', 'MEL', 'NV']
    idx = np.argmax(predictions)
    print(f"Diagnosis: {classes[idx]} | Confidence: {predictions[0][idx]*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image.jpg>")
    else:
        predict_single_image(sys.argv[1])

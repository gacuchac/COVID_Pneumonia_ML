import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np 

import tensorflow as tf

model_name = 'Models/VGG16_final_model/VGG16_model.h5'
model = tf.keras.models.load_model(model_name)

st.write("""
         # COVID-19 and Viral Pneumonia Predictions
         """
         )
st.write("This is a simple image classification web app to predict if a patient has COVID-19, Viral Pneumonia or is Healthy")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224,224)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    st.write(image)
    image = np.asarray(image)
    st.write(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.
        
    img_reshape = img_resize[np.newaxis,...]
    
    prediction = model.predict(img_reshape)
        
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    st.text("Probability (0: COVID, 1: NORMAL, 2: Viral Pneumonia")
    st.write(prediction)
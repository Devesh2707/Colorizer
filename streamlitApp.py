from utils import download_weights, predict
from generator import load_weights
import numpy as np
import cv2
import torch
import io
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

device = "cpu"

def main():
    download_weights()
    st.title("Black&White photo colorizer")
    file = st.file_uploader("upload a black&white photo", type=['jpg','png','jpeg'])
    image_size = int(st.number_input("Provide image size (note: larger images will take more time and will not show good results too)", min_value=128.0, max_value=1024.0, value = 224.0))
    model = load_weights(device=device, image_size=image_size)
    if file is not None:
        img_stream = io.BytesIO(file.read())
        img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
        st.image(img, width=image_size)
        color_image = st.button("Color it!")
        if color_image is True:
            color_img = predict(img, device, image_size, model)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            st.image(color_img, width = image_size)    
    
if __name__ == "__main__":
    main()
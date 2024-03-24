#Function to preprocess image for prediction
from skimage import io, color, exposure, transform, util
import streamlit as st
import joblib
import numpy as np
import matplotlib as plt

# Load the Random Forest tuned model and the scaler 
#model_path = "RF_model.joblib" 
#scaler_path = "scaler.joblib"
#model = joblib.load(model_path)
#scaler = joblib.load(scaler_path) 

model_path = "SVC_model.joblib" 
scaler_path = "scaler.joblib"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path) 

# Print the model itself and its parameters 
print(model)
#print("Model parameters:", model.get_params())

def preprocess_image(image_path):
    # Read the image
    image = io.imread(image_path)
    
    # If the image is RGBA, convert to RGB
    if image.ndim == 3 and image.shape[2] == 4:
        image = color.rgba2rgb(image)
    
    # Convert to grayscale
    image_gray = color.rgb2gray(image)
    
    # Here we tested for the best threshold value
    thresh = 0.5
    
    # Apply thresholding to create a binary image
    binary_image = image_gray > thresh
    
    # Invert colors of the binary image
    # Since the image is binary (True/False), we can invert it using the ~ operator directly
    image_inverted = ~binary_image
    
    # Resize the image
    image_resized = transform.resize(image_inverted, (28, 28), anti_aliasing=False, mode='reflect')
    
    # Convert the resized image to 8-bit unsigned integer format (0 to 255 range)
    image_ubyte = util.img_as_ubyte(image_resized)

    # Flatten the image
    image_flattened = image_ubyte.flatten()
    
    # Reshape for a single sample
    image_reshaped = image_flattened.reshape(1, -1)
    
    return image_reshaped

st.title('MNIST Digit Predictor')

use_camera = st.checkbox("Enable Camera", value=False)
uploaded_file = st.file_uploader("Upload an image of a handwritten digit", type=["png", "jpg", "jpeg"], disabled=use_camera)
img_file_buffer = st.camera_input("Take a picture") if use_camera else None

# Process uploaded file or camera input
image_to_process = None
if uploaded_file is not None:
    # Save the uploaded file to a temporary file and pass that path to preprocess_image
    with open("temp_image", "wb") as f:
        f.write(uploaded_file.getvalue())
    image_to_process = "temp_image"
elif img_file_buffer is not None:
    # Similar handling for camera input
    with open("temp_camera_image", "wb") as f:
        f.write(img_file_buffer.getvalue())
    image_to_process = "temp_camera_image"

if image_to_process:
    st.image(image_to_process, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image_to_process)

    # Apply scaler 
    processed_image_transformed = scaler.transform(processed_image) 
    
    # Predict the digit
    st.write("Classifying...")
    prediction = model.predict(processed_image)
    st.write(f'The handwritten digit is likely a: {prediction[0]}')



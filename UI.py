from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pickle
import pandas as pd
import socket

# Socket fix for Windows
socket.socket._bind = socket.socket.bind
def socket_bind(self, *args, **kwargs):
    self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return socket.socket._bind(self, *args, **kwargs)
socket.socket.bind = socket_bind

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load all trained models
tumor_model = load_model('brain_tumor_model.h5')

# Load confirmation model
try:
    with open("confirmation_model.pkl", "rb") as file:
        confirmation_model = pickle.load(file)
    if not hasattr(confirmation_model, 'predict'):
        raise AttributeError("Confirmation model doesn't have predict method")
except Exception as e:
    print(f"Error loading confirmation model: {e}")
    confirmation_model = None

# Load food suggestion model and encoders
try:
    with open("food_model.pkl", "rb") as file:
        food_model = pickle.load(file)
    with open("tumor_encoder.pkl", "rb") as file:
        le_tumor = pickle.load(file)
    with open("food_encoder.pkl", "rb") as file:
        le_food = pickle.load(file)
except Exception as e:
    print(f"Error loading food suggestion models: {e}")
    food_model = None

# Class names for tumor detection
tumor_class_names = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_tumor(img_path):
    """Predict tumor type from MRI image"""
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = tumor_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])  # Keep as decimal (0-1)
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error in tumor prediction: {e}")
        return None, 0

def predict_confirmation(tumor_class, confidence, wbc, rbc, age, symptoms):
    """Predict tumor confirmation using clinical data"""
    if confirmation_model is None:
        return False, 0.0
        
    try:
        input_features = np.array([[
            tumor_class, confidence, wbc, rbc, age,
            symptoms['headache'], symptoms['nausea'],
            symptoms['seizure'], symptoms['vision_blur']
        ]])
        
        if hasattr(confirmation_model, 'predict_proba'):
            confirmation_prob = confirmation_model.predict_proba(input_features)[0][1]
            confirmed = confirmation_model.predict(input_features)[0]
            return bool(confirmed), confirmation_prob
        else:
            confirmed = confirmation_model.predict(input_features)[0]
            return bool(confirmed), 1.0 if confirmed else 0.0
            
    except Exception as e:
        print(f"Error in confirmation prediction: {e}")
        return False, 0.0

def suggest_foods(tumor_type, age):
    """Get food suggestions based on tumor type and age"""
    if food_model is None:
        return "No food suggestions available"
    
    try:
        tumor_encoded = le_tumor.transform([tumor_type])[0]
        food_encoded = food_model.predict([[tumor_encoded, age]])[0]
        return le_food.inverse_transform([food_encoded])[0]
    except Exception as e:
        print(f"Error in food suggestion: {e}")
        return "Error generating food suggestions"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Get tumor prediction
                tumor_class, confidence = predict_tumor(filepath)
                if tumor_class is None:
                    return render_template('error.html', message="Tumor prediction failed")
                
                tumor_type = tumor_class_names.get(tumor_class, "Unknown Tumor Type")
                
                # Get clinical parameters
                try:
                    age = int(request.form.get('age', 30))
                    wbc = float(request.form.get('wbc', 7000))
                    rbc = float(request.form.get('rbc', 5.0))
                    
                    symptoms = {
                        'headache': int(request.form.get('headache', 0)),
                        'nausea': int(request.form.get('nausea', 0)),
                        'seizure': int(request.form.get('seizure', 0)),
                        'vision_blur': int(request.form.get('vision_blur', 0))
                    }
                except ValueError as e:
                    return render_template('error.html', message=f"Invalid clinical data: {e}")
                
                # Get confirmation prediction
                confirmed, confirmation_prob = predict_confirmation(
                    tumor_class, confidence, wbc, rbc, age, symptoms
                )
                
                # Get food suggestions
                food_recommendations = suggest_foods(tumor_type, age)
                
                # Convert to percentages for display
                display_confidence = round(confidence * 100, 2)
                display_confirmation_prob = round(confirmation_prob * 100, 2)
                
                return render_template('result.html', 
                                    image_file=filename,
                                    prediction=tumor_type,
                                    confidence=display_confidence,
                                    age=age,
                                    wbc=wbc,
                                    rbc=rbc,
                                    symptoms=symptoms,
                                    confirmed='Yes' if confirmed else 'No',
                                    confirmation_prob=display_confirmation_prob,
                                    food_recommendations=food_recommendations)
            
            except Exception as e:
                return render_template('error.html', message=f"An error occurred: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=False
    )

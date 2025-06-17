import os
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
import tensorflow as tf

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sports_classifier.keras')
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Badminton', 'Cricket', 'Karate', 'Soccer', 'Swimming', 'Tennis', 'Wrestling']  # <-- replace with your actual classes

def predict_view(request):
    prediction = None
    confidence = None
    uploaded_image_url = None

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        uploaded_image_url = file.name  # just for displaying name (we won't save file)

        img = Image.open(file).resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = MODEL.predict(img)[0]
        prediction = CLASS_NAMES[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100

    return render(request, 'home.html', {
        'prediction': prediction,
        'uploaded_image_url': uploaded_image_url
    })
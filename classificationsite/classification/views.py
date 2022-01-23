from django.shortcuts import render, redirect
from django.http import HttpResponse, request
from .forms import Records
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
# Create your views here.


def index(request):
    return redirect('home/')

def home(request):
    if request.method == "POST":
        form = Records(request.POST)
        img = request.FILES.get('scan')
        image = Image.open(img)
        image.save('temp.jpg')
        outcome = check()
        return render(request, "classification/home.html", {"form": form, "check": outcome, "url": '/Users/teerthapenumatcha/Desktop/classification/classificationsite/temp.jpg'})

    else:
        form = Records()
    return render(request, "classification/home.html", {"form": form})


def check():
    path = 'temp.jpg'
    predictions = []
    img = image.load_img(path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    scan = np.vstack([x])
    models = []
    for model in os.listdir('/Users/teerthapenumatcha/Desktop/classification/fold_models'):
        models.append(tf.keras.models.load_model('/Users/teerthapenumatcha/Desktop/classification/fold_models/' + model))
    for model in models:
        predictions.append(model.predict(x))
    overall = np.array(predictions).mean() * 100
    overall = round(overall, 2)
    print(overall)
    if overall > 50:
        return f"Malignent with {overall}% confidence"
    else:
        return f"Benign with {100 - overall}% confidence"



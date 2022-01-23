from tkinter import *
from typing import overload
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os

root = Tk()
root.title("Brain Tumor Analysis")
root.geometry("550x300")

result = ""

def check():
    path = 'temp.jpg'
    predictions = []
    img = image.load_img(path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    scan = np.vstack([x])
    models = []
    for model in os.listdir('fold_models/'):
        models.append(tf.keras.models.load_model('fold_models/' + model))
    for model in models:
        predictions.append(model.predict(x))
    overall = np.array(predictions).mean() * 100
    overall = round(overall, 2)
    print(overall)
    if overall > 50:
        output.config(text=f"Malignent with {overall}% confidence")
    else:
        output.config(text=f"Benign with {100 - overall}% confidence")

def open_img():
    x = openfilename()
    
    img = Image.open(x)
    pic = img.save("temp.jpg")

    img = img.resize((200, 200), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image = img)
    panel.image = img
    panel.grid(row = 2)

    check()



def openfilename():
    filename = filedialog.askopenfilename(title ='open')
    return filename



btn = Button(root, text ='Check', command = open_img).grid(row = 1, column = 0)

output = Label(root, text="")
label = Label(root, text="Input MRI scan to test Brain Tumor Lethality")
label.grid(row=0,column=0)
output.grid(row=2, column=4)

root.mainloop()
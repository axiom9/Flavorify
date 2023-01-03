import sys
sys.path.insert(0, '/Users/anasputhawala/Desktop/Winterproj')
import tensorflow as tf
import numpy as np
import re
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__)

MODEL_PATH = "model_weights/"

mapping = { '❤':'0' , '😍':'1' , '😂':'2' , '💕':'3' , '🔥':'4' , '😊':'5' , 
            '😎':'6' , '✨':'7' , '😜':'8' , '😘':'9' , '☀':'10', '📸':'11' , 
            '😉':'12','💯':'13','🇺🇸':'14' ,'🎄':'15' }

mapping_rev = {int(v): k for k, v in mapping.items()}

# Loading model
model = tf.keras.models.load_model(MODEL_PATH)

def model_predict(sentence:str):
    formatted = re.split(r'[,;!?.:]', sentence)
    preds = np.argmax(model.predict(formatted), axis=-1)
    sout=""
    for idx, sent in enumerate(formatted):
        sout = sout+sent+" "+mapping_rev[preds[idx]]
    return sout

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form.values()
    return render_template('index.html', prediction_text=f'{model_predict(next(sentence))}')

if __name__ == "__main__":
    app.run(debug=True)

import tensorflow as tf
import numpy as np
import re
from flask import Flask, request, render_template

# Define a flask app
app = Flask(__name__)

#Model path
MODEL_PATH = "model_weights/"

#mappings for emojis
mapping = { 'β€':'0' , 'π':'1' , 'π':'2' , 'π':'3' , 'π₯':'4' , 'π':'5' , 
            'π':'6' , 'β¨':'7' , 'π':'8' , 'π':'9' , 'β':'10', 'πΈ':'11' , 
            'π':'12','π―':'13','πΊπΈ':'14' ,'π':'15' }

mapping_rev = {int(v): k for k, v in mapping.items()}

# Loading model
model = tf.keras.models.load_model(MODEL_PATH)

#model pred function
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
    app.run()

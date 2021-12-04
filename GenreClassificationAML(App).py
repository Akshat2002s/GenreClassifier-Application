from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
import librosa
import os
from tensorflow.keras.models import Sequential, save_model, load_model
seed = 12
np.random.seed(seed)

app= Flask(__name__)

genre=pd.DataFrame(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop',
'reggae', 'rock'])

lb = LabelEncoder()
genre = to_categorical(lb.fit_transform(genre))

model=load_model('C:/Users/aksha/AMLModel(2).h5')

def model_predict(file_path, model):
  ans = pd.DataFrame(columns=['feature'])
  signal, sr = librosa.load(file_path)
  pred=np.array([])
  mfccs=np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
  pred=np.hstack((pred, mfccs))

  stft=np.abs(librosa.stft(signal))
  chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
  pred=np.hstack((pred, chroma))

  mel=np.mean(librosa.feature.melspectrogram(signal, sr=sr).T,axis=0)
  pred=np.hstack((pred, mel))

  ans.loc[0]=[pred]
  final_ans=pd.DataFrame(ans['feature'].values.tolist())

  predictions = model.predict(final_ans)
  predictions=predictions.argmax(axis=1)
  predictions = predictions.astype(int).flatten()
  predictions = (lb.inverse_transform((predictions)))
  return np.array_str(predictions)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        
        preds = model_predict(file_path, model)
        
        return preds
    return None

if __name__ == '__main__':
    app.run(debug=True)

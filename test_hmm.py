import warnings
import argparse 
import os
import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm 
import librosa
import matplotlib.pyplot as plt
import scipy.fft as sc
from librosa.feature import mfcc
import librosa.display

warnings.filterwarnings('ignore')

class HMMTrainer(object):
   def __init__(self, model_name='GaussianHMM', n_components=6):
     self.model_name = model_name
     self.n_components = n_components

     self.models = []
     if self.model_name == 'GaussianHMM':
        self.model=hmm.GaussianHMM(n_components=6)
     else:
        print("Please choose GaussianHMM")

   def train(self, X):
       self.models.append(self.model.fit(X))

   def get_score(self, input_data):
       return self.model.score(input_data)

def plot_amplitude(audio_file):
    y, sr =  librosa.load(audio_file)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))
    fig.suptitle(audio_file)
    ax1.set_title("Audio")
    ax1.plot(y)
    ax2.set_title("fft")
    ax2.plot(sc.fft(y))
    plt.show()

def plot_melspectrogram(audio_file):
    y, sr =  librosa.load(audio_file)
    fig, ax = plt.subplots()
    M = librosa.feature.melspectrogram(y=y, sr=sr)
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, ax=ax)
    ax.set(title=audio_file)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

input_folder = "C:/Users/Mi/Desktop/project/audio/"

labels = os.listdir(input_folder)



hmm_models = []
for dirname in os.listdir(input_folder):
    subfolder = os.path.join(input_folder, dirname)
    # print(subfolder)  
    if not os.path.isdir(subfolder): 
         continue
    label = subfolder[subfolder.rfind('/') + 1:]
    X = np.array([])
    for filename in [f for f in os.listdir(subfolder) if f.endswith('.wav')][:-1]:
        filepath = os.path.join(subfolder, filename)
        sampling_freq, audio = librosa.load(filepath)            
        mfcc_features = mfcc(sampling_freq, audio)
        if len(X) == 0:
            X = mfcc_features[:,:15]
        else:
            X = np.append(X, mfcc_features[:,:15], axis=0)            
    #print('X.shape =', X.shape)
    hmm_trainer = HMMTrainer()
    hmm_trainer.train(X)
    hmm_models.append((hmm_trainer, label))
    hmm_trainer = None

input_files = [
'C:/Users/Mi/Desktop/project/audio/pineapple/pineapple15.wav',
'C:/Users/Mi/Desktop/project/audio/orange/orange15.wav',
'C:/Users/Mi/Desktop/project/audio/apple/apple15.wav',
'C:/Users/Mi/Desktop/project/audio/kiwi/kiwi15.wav',
'C:/Users/Mi/Desktop/project/audio/lime/lime15.wav',
'C:/Users/Mi/Desktop/project/audio/banana/banana15.wav',
'C:/Users/Mi/Desktop/project/audio/peach/peach15.wav'
]

# plot_amplitude(input_files[0])
#plot_melspectrogram(input_files[0])
print(labels)

for input_file in input_files:
    sampling_freq, audio = librosa.load(input_file)

    # Extract MFCC features
    mfcc_features = mfcc(sampling_freq, audio)
    mfcc_features = mfcc_features[:,:15]

    scores=[]
    for item in hmm_models:
        hmm_model, label = item    
        score = hmm_model.get_score(mfcc_features)
        scores.append(score)
    # print("\n", scores)

    index = np.array(scores).argmax()
    print("File read:", input_file)
    print("Predicted:", hmm_models[index][1])
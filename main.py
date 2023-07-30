# Imports and setup
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from glob import glob
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle
import os
from pathlib import Path
sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
# Getting all the audio files together for training
audio_files = glob("data/train_data/*.wav")
# Converting the audio into numbers
data = {
    "itemid": [],
    "data": []
}
for file in audio_files:
    y, sr = librosa.load(file)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    file_name = Path(file).stem
    
    data["itemid"].append(file_name)
    data["data"].append(ps.flatten())
  
# Inital data collection  
X = pd.DataFrame(data)
y = pd.read_csv("data/train_labels.csv")
# Ordering the data so that the audio data lines up with the csv train labels
merged = X.merge(y, on="itemid", how="inner")
# getting the data after its been ordered
X = merged["data"].to_frame()
y = merged["hasbird"]
# Formatting the data for the model
X_padded = pad_sequences(X["data"], padding='post', dtype='float32')
X_padded_df = pd.DataFrame(X_padded)
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_padded_df, y, test_size=0.2)

# Training model and predicting

# Classification
# KNeighborsClassifier()
# DecisionTreeClassifier()

# Regression
# KNeighborsRegressor

model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Training model accuracy:", acc)

audio_files = glob("data/test_data/*.wav")
# Converting the audio into numbers
data = {
    "itemid": [],
    "data": []
}
for file in audio_files:
    y, sr = librosa.load(file)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    file_name = Path(file).stem
    
    data["itemid"].append(file_name)
    data["data"].append(ps.flatten())
  
# Inital data collection  
X = pd.DataFrame(data)
# Ordering the data so that the audio data lines up with the csv train labels
# getting the data after its been ordered
X = X["data"].to_frame()
# Formatting the data for the model
X_padded = pad_sequences(X["data"], padding='post', dtype='float32')
X_padded_df = pd.DataFrame(X_padded)
# Train test split

# Training model and predicting

# Classification
# KNeighborsClassifier()
# DecisionTreeClassifier()

# Regression
# KNeighborsRegressor

y_pred = model.predict(X)

# Accuracy
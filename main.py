import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# get the train audio files
audio_files = glob("data/train_data/*.wav")

# Making audio files usable data
X_data = [
    {
        "itemid": Path(file).stem,
        "data": librosa.feature.melspectrogram(y=y, sr=sr)
    }
    for file in audio_files
    for y, sr in [librosa.load(file)]
]

X = pd.DataFrame(X_data)

y = pd.read_csv("data/train_labels.csv")

merged = X.merge(y, on="itemid", how="inner")

X = merged["data"]
y = merged["hasbird"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

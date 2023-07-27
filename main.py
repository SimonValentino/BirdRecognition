import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# get the train audio files
train = glob("../data/train_data/*.wav")

# loading the audio files with librosa (converting to numbers)
loaded_train = []
for audio_file in train:
    loaded_train.append(librosa.load(audio_file)[0])

# applying fourier transform on data to get Hz over time
transformed_train = []
for data in loaded_train:
    D = librosa.stft(data)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    transformed_train.append(S_db)

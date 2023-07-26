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
audio_files = glob("./data/train_data/*.wav")
data, sr = librosa.load(audio_files[80])
pd.Series(data).plot(figsize=(10, 5),
                     lw=1,
                     title="Raw Audio Example",
                     color=color_pal[0])
plt.show()
#print(ipd.Audio(audio_files[80]))
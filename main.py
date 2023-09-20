import librosa
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

rawData, integerdata = librosa.load(r"C:\Users\nicol\Music\Investigación\Papel celulosa\5x5cm\Papel_celulosa_5x5cm_usos_0_distancia_10cm_192000Hz.wav")

print(rawData)
print(rawData.shape)
print(integerdata)

pandas.Series(rawData).plot(figsize = (10, 5), lw = 1, title = "Representación de ejemplo", color=color_pal[0])
plt.show()

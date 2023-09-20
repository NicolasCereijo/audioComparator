import librosa
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

rawData, integerData = librosa.load(r"C:\Users\nicol\Music\Investigación\Papel celulosa\5x5cm\Papel_celulosa_5x5cm_usos_0_distancia_10cm_192000Hz.wav")

print(rawData)
print(rawData.shape)
print(integerData)

pandas.Series(rawData).plot(figsize = (10, 5), lw = 1, title = "Representación de ejemplo", color=color_pal[0])
plt.show()

pandas.Series(rawData[30000:30500]).plot(figsize = (10, 5), lw = 1, title = "Representación de ejemplo aumentada", color = color_pal[1])
plt.show()

# Representación del espectrograma
STFT = librosa.stft(rawData)
decibels = librosa.amplitude_to_db(numpy.abs(STFT), ref = numpy.max)

figure, axes = plt.subplots(figsize=(10, 5))
img = librosa.display.specshow(decibels, x_axis = "time", y_axis = "log", ax = axes)
axes.set_title("Ejemplo de espectrograma", fontsize = 20)
figure.colorbar(img, ax = axes, format = f"%0.2f")
plt.show()

# Representación del espectrograma Mel
S = librosa.feature.melspectrogram(y = rawData, sr = integerData, n_mels =128 * 2, )
decibelsMel = librosa.amplitude_to_db(S, ref = numpy.max)

figure, axes = plt.subplots(figsize = (10, 5))
image = librosa.display.specshow(decibelsMel, x_axis = "time", y_axis = "log", ax = axes)
axes.set_title('Ejemplo de espectrograma Mel', fontsize = 20)
figure.colorbar(image, ax = axes, format = f'%0.2f')
plt.show()

import librosa
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
from itertools import cycle
# https://github.com/gudgud96/frechet-audio-distance
from frechet_audio_distance import FrechetAudioDistance
# torchvision needed

seaborn.set_theme(style = "white", palette = None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

audioSignal, sampleRate = librosa.load(r"C:\Users\ganst\Music\Investigación\Papel celulosa\5x5cm\Papel_celulosa_5x5cm_usos_0_distancia_10cm_192000Hz.wav")
audioSignal2, sampleRate2 = librosa.load(r"C:\Users\ganst\Music\Investigación\Papel celulosa\20x20cm\Papel_celulosa_20x20cm_usos_0_distancia_10cm_192000Hz.wav")

print(audioSignal)
print(audioSignal.shape)
print(sampleRate)

# Imagen del sonido
pandas.Series(audioSignal).plot(figsize = (10, 5), lw = 1, title = "Representación de ejemplo", color = color_pal[0])
pandas.Series(audioSignal2).plot(figsize = (10, 5), lw = 1, title = "Representación de ejemplo", color = color_pal[1])
plt.show()

# Imagen del sonido ampliada
pandas.Series(audioSignal[30000:30500]).plot(figsize = (10, 5), lw = 1, title ="Representación de ejemplo aumentada", color = color_pal[1])
plt.show()

# Representación del espectrograma
STFT = librosa.stft(audioSignal)
decibels = librosa.amplitude_to_db(numpy.abs(STFT), ref = numpy.max)

figure1, axes1 = plt.subplots(figsize = (10, 5))
image1 = librosa.display.specshow(decibels, x_axis = "time", y_axis = "log", ax = axes1)
axes1.set_title("Ejemplo de espectrograma", fontsize = 20)
figure1.colorbar(image1, ax = axes1, format = f"%0.2f")
plt.show()

# Representación del espectrograma Mel
melSpectrogram = librosa.feature.melspectrogram(y = audioSignal, sr = sampleRate, n_mels = 128 * 2, )
decibelsMel = librosa.amplitude_to_db(melSpectrogram, ref = numpy.max)

figure2, axes2 = plt.subplots(figsize = (10, 5))
image2 = librosa.display.specshow(decibelsMel, x_axis = "time", y_axis = "log", ax = axes2)
axes2.set_title('Ejemplo de espectrograma Mel', fontsize = 20)
figure2.colorbar(image2, ax = axes2, format = f'%0.2f')
plt.show()

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name = "vggish",
    sample_rate = 16000,
    use_pca = False,
    use_activation = False,
    verbose = False
)
# to use `PANN`
frechet = FrechetAudioDistance(
    model_name = "pann",
    sample_rate = 16000,
    use_pca = False,
    use_activation = False,
    verbose = False
)
# to use `CLAP`
frechet = FrechetAudioDistance(
    model_name = "clap",
    sample_rate = 48000,
    submodel_name = "630k-audioset",  # for CLAP only
    verbose = False,
    enable_fusion = False,            # for CLAP only
)

print(frechet.score("C:/Users/ganst/Music/Investigación/audio1", "C:/Users/ganst/Music/Investigación/audio2", dtype = "float32"))

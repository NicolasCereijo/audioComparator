import librosa
import pandas
import matplotlib.pyplot as plt
import seaborn
from itertools import cycle
# Repository: https://github.com/gudgud96/frechet-audio-distance
from frechet_audio_distance import FrechetAudioDistance
# Package torchvision needed

# PANN model
frechet = FrechetAudioDistance(
    model_name="pann",
    # Sample rate must be 8000, 16000 or 32000
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False
)

seaborn.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

audioSignal1, sampleRate1 = librosa.load(r"/home/nico/GitHub/audioComparator/audioPrueba1/Papel_celulosa_5x5cm_usos_0_distancia_10cm_192000Hz.wav")
audioSignal2, sampleRate2 = librosa.load(r"/home/nico/GitHub/audioComparator/audioPrueba2/Papel_celulosa_20x20cm_usos_3_distancia_100cm_192000Hz.wav")

print("Frechet distance: ", frechet.score("/home/nico/GitHub/audioComparator/audioPrueba1",
                                          "/home/nico/GitHub/audioComparator/audioPrueba2", dtype="float32"))
print("Sample rate model: ", frechet.sample_rate)
print("Sample rate audio 1: ", sampleRate1)
print("Sample rate audio 2: ", sampleRate2)

# Sound image
pandas.Series(audioSignal1).plot(figsize=(10, 5), lw=1, title="Representación de ejemplo", color=color_pal[0])
pandas.Series(audioSignal2).plot(figsize=(10, 5), lw=1, title="Representación de ejemplo", color=color_pal[1])
plt.show()

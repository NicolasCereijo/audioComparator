import librosa
import pandas
import matplotlib.pyplot as plt
import seaborn
from itertools import cycle
import glob
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

referenceFilesPath = "referenceFiles/"
fileToComparePath = "fileToCompare/"
referenceFiles = glob.glob(referenceFilesPath + "*.wav")
fileToCompare = glob.glob(fileToComparePath + "*.wav")
audioSignal1, sampleRate1 = librosa.load(referenceFiles[0])
audioSignal2, sampleRate2 = librosa.load(fileToCompare[0])

print("Frechet distance: ", frechet.score(referenceFilesPath, fileToComparePath, dtype="float32"))
print("Sample rate model: ", frechet.sample_rate)
print("Sample rate audio 1: ", sampleRate1)
print("Sample rate audio 2: ", sampleRate2)

seaborn.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Sound image
pandas.Series(audioSignal1).plot(figsize=(10, 5), lw=1, title="", color=color_pal[0])
pandas.Series(audioSignal2).plot(figsize=(10, 5), lw=1, title="Audios to compare", color=color_pal[1])
plt.show()

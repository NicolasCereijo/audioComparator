import glob
from itertools import cycle

import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from frechet_audio_distance import FrechetAudioDistance

# Frechet repository: https://github.com/gudgud96/frechet-audio-distance
# Package torchvision needed

# PANN model
frechet = FrechetAudioDistance(
    model_name="pann",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False
)

reference_path = "referenceFiles/"
compare_path = "fileToCompare/"
reference_files = glob.glob(reference_path + "*.wav")
compare_files = glob.glob(compare_path + "*.wav")


def load_audio(file_path):
    audio_signal, sample_rate = librosa.load(file_path)
    return audio_signal, sample_rate


def plot_audio_series(audio_signal, title, color):
    pd.Series(audio_signal).plot(figsize=(10, 5), lw=1, title=title, color=color)


def main():
    audio_signal1, sample_rate1 = load_audio(reference_files[0])
    audio_signal2, sample_rate2 = load_audio(compare_files[0])

    print("Frechet distance: ", frechet.score(reference_path, compare_path, dtype="float32"))
    print("Sample rate model: ", frechet.sample_rate)
    print("Sample rate audio 1: ", sample_rate1)
    print("Sample rate audio 2: ", sample_rate2)

    sns.set_theme(style="white", palette=None)
    color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    # Sound image
    plot_audio_series(audio_signal1, "", color_pal[0])
    plot_audio_series(audio_signal2, "Audios to compare", color_pal[1])
    plt.show()


if __name__ == "__main__":
    main()

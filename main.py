import glob
from itertools import cycle

import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from frechet_audio_distance import FrechetAudioDistance
from sklearn.metrics.pairwise import cosine_similarity

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


def calculate_mfcc(audio_signal, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=20)
    return mfcc


def compare_mfcc(audio_signal1, audio_signal2, sample_rate1, sample_rate2):
    mfcc1 = calculate_mfcc(audio_signal1, sample_rate1)
    mfcc2 = calculate_mfcc(audio_signal2, sample_rate2)

    # Ensure that both vectors have the same length
    min_length = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_length]
    mfcc2 = mfcc2[:, :min_length]

    # Reshape the MFCC matrices to vectors for cosine similarity calculation
    mfcc1_vector = mfcc1.reshape(-1)
    mfcc2_vector = mfcc2.reshape(-1)

    # Calculate cosine similarity
    similarity_percentage = cosine_similarity([mfcc1_vector], [mfcc2_vector])[0][0] * 100
    return similarity_percentage


def plot_audio_series(audio_signal, title, color):
    pd.Series(audio_signal).plot(figsize=(10, 5), lw=1, title=title, color=color)


def main():
    audio_signal1, sample_rate1 = load_audio(reference_files[0])
    audio_signal2, sample_rate2 = load_audio(compare_files[0])

    print("Sample rate model: ", frechet.sample_rate)
    print("Sample rate audio 1: ", sample_rate1)
    print("Sample rate audio 2: ", sample_rate2)
    print("Frechet distance: ", frechet.score(reference_path, compare_path, dtype="float32"))

    sns.set_theme(style="white", palette=None)
    color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    # Compare MFCC and print the similarity percentage
    similarity_percentage = compare_mfcc(audio_signal1, audio_signal2, sample_rate1, sample_rate2)
    print("Similarity Percentage (MFCC): {:.2f}%".format(similarity_percentage))

    # Sound image
    plot_audio_series(audio_signal1, "", color_pal[0])
    plot_audio_series(audio_signal2, "Audios to compare", color_pal[1])
    plt.show()


if __name__ == "__main__":
    main()

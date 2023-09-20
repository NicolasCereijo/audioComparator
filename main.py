import librosa
import pandas
import matplotlib.pyplot as plt

rawData, integerdata = librosa.load(r"C:\Users\nicol\Music\Investigación\Papel celulosa\5x5cm\Papel_celulosa_5x5cm_usos_0_distancia_10cm_192000Hz.wav")

print(rawData)
print(rawData.shape)
print(integerdata)

pandas.Series(rawData).plot(figsize = (10, 5), lw = 1, title = "Representación de ejemplo")
plt.show()

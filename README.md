<h1 align="center">Audio comparator, based on Fréchet distance</h1>
<h3 align="left">🈳 🇪🇦 Este repositorio utiliza la librería "frechet_audio_distance" (enlace más abajo) para realizar comparaciones entre ficheros de audio ".wav".
  Dicha librería implementa tres modelos diferentes para realizar los cálculos, VGGish, PANN y CLAP, siendo el segundo el utilizado en este programa.
  El modelo realiza una comparación del audio refenrecia con el audio a comparar y devuelve un resultado entre 0 y 1, siendo 0 para audios completamente iguales y 1 para audios completamente diferentes.
  El programa solo utiliza un audio como referencia, pero las listas implementadas permiten ampliarlo para hacer multitud de comparaciones y una media global.</h3>
<br/>
<h3 align="left">🈳 🇬🇧 This repository uses the "frechet_audio_distance" library (link below) to perform comparisons between ".wav" audio files.
   This library implements three different models to perform the calculations, VGGish, PANN and CLAP, the second being the one used in this program.
   The model performs a comparison of the reference audio with the audio to be compared and returns a result between 0 and 1, with 0 for completely identical audios and 1 for completely different audios.
   The program only uses one audio as a reference, but the implemented lists allow it to be expanded to make a multitude of comparisons and a global average.</h3>
<br/>
- 📝 Library [frechet_audio_distance](https://github.com/gudgud96/frechet-audio-distance)
- 📝 Model documentation [PANN](https://arxiv.org/abs/1912.10211)

import librosa
import numpy as np
import tempfile


def preprocess_audio(audio_bytes):

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(audio_bytes)
        audio_path = temp.name

    signal, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc
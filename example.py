import chirplet as ch
import librosa

chirps = ch.FCT()
audio, sr = librosa.load('audio/sa2.wav')

fct = chirps.compute(audio)

print(fct)

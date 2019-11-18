# Pytorch-audio_feature
Audio feature extraction in Pytorch module.

## To test melspectrogram
```
import torch
from audio_feature import STFT
from audio_feature import melspectrogram
import numpy as np
import librosa 
import matplotlib.pyplot as plt
# from spec import Melspectrogram

audio = librosa.load(librosa.util.example_audio_file(),sr=16000)[0]
audio = audio[:16000]
device = 'cuda'
filter_length = 1024
hop_length = 512
win_length = 1024 # doesn't need to be specified. if not specified, it's the same as filter_length
window = 'hann'
librosa_stft = librosa.stft(audio, n_fft=filter_length, hop_length=hop_length, window=window)
_magnitude = np.abs(librosa_stft)

y = audio
D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D,n_mels = 128,sr=16000)
# melspec = librosa.feature.melspectrogram(_magnitude**2, sr=44100,
#                                     n_fft=1024, 
#                                     hop_length=512, 
#                                     n_mels = 128) 

audio = torch.FloatTensor(audio)
audio = audio.unsqueeze(0)
audio = audio.to(device)

stft = STFT(
    filter_length=filter_length, 
    hop_length=hop_length, 
    win_length=win_length,
    window=window
).to(device)
# spect = Melspectrogram(hop=512, n_fft=1024, window=None, sr=44100).to(device)
# ss = spect.forward(audio.unsqueeze(0))
spect = melspectrogram(n_fft=1024, sr=16000).to(device)
magnitude, phase = stft.transform(audio)
ss = spect.forward(magnitude.to(device))
# ss = librosa.feature.melspectrogram(S=magnitude.squeeze(0).to('cpu'),n_fft=1024,sr=44100)
# print(ss.shape)
plt.figure(figsize=(6, 6))
plt.subplot(211)
plt.title('PyTorch STFT magnitude')
plt.xlabel('Frames')
plt.ylabel('FFT bin')
# plt.imshow(20*np.log10(1+magnitude[0].cpu().data.numpy()), aspect='auto', origin='lower')
plt.imshow(20*np.log10(1+ss[0].cpu().data.numpy()), aspect='auto', origin='lower')
plt.subplot(212)
plt.title('Librosa STFT magnitude')
plt.xlabel('Frames')
plt.ylabel('FFT bin')
# plt.imshow(20*np.log10(1+_magnitude), aspect='auto', origin='lower')
plt.imshow(20*np.log10(1+S), aspect='auto', origin='lower')
plt.tight_layout()
plt.savefig('images/stft.png')


```

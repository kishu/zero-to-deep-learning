import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

#
# Images
#
img=Image.open('../data/iss.jpg')
imgarray = np.asarray(img)
#print(imgarray.shape)
#print(imgarray.ravel().shape)

#
# Sound
#
from scipy.io import wavfile
from IPython.display import Audio
rate, snd = wavfile.read(filename='../data/sms.wav')
Audio(data=snd, rate=rate)
print(len(snd))
print(snd)
#plt.plot(snd)
#plt.show()
_ = plt.specgram(snd, NFFT=1024, Fs=44100)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.show()


import sounddevice as sd
from scipy.io.wavfile import write
from scipy import signal
import matplotlib.pyplot as plt
import librosa

fs = 48000
duration_of_record = 5


def recording(duration_of_record, fs):
    record = sd.rec(int(duration_of_record * fs), samplerate=fs, channels=2)
    sd.wait()
    return record


def write_to_wav(FILEPATH, record, fs):
    return write(FILEPATH, fs, record)


def louder(record, multiplier):
    return record*multiplier


def quieter(record, divider):
    return record/divider


def pitch_shifting(FILE_PATH, fs, n_steps):
    y, sr = librosa.load(FILE_PATH, sr=fs)
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)


#fir filter
def low_pass_fir_filter(N, fs, fc, record):
    b = signal.firwin(N, fc, fs=fs, pass_zero=True, window='hamming')
    return signal.lfilter(b, [1.0], record)


#iir filter
def low_pass_iir_filter(N, record):
    b, a = signal.butter(N, 0.5, btype='low', analog=False)
    return signal.lfilter(b, a, record)


def play_record(record, fs):
    return sd.play(record, fs), sd.wait()


FILEPATH = 'output.wav'
voice_record = recording(duration_of_record, fs)
write_to_wav(FILEPATH, voice_record, fs)
voice_shifted = pitch_shifting(FILEPATH, fs, 12)
voice_fir_filtered = low_pass_fir_filter(5, fs, 70, voice_record)
voice_iir_filtered = low_pass_iir_filter(5, voice_record)
voice_louder = louder(voice_record, 4)
voice_quieter = quieter(voice_record, 2)


plt.figure(figsize=(10, 10))

plt.subplot(131)
plt.plot(voice_record, color='g')
plt.ylabel('Amplitude')
plt.xlabel('fs')
plt.ylim(-0.5, 0.5)
plt.title('Original')


'''
#Plot lowpass fir filter amplitude
plt.subplot(132)
plt.plot(FIR_FilteredSignal, color='r')
plt.ylabel('Amplitude')
plt.xlabel('fs')
plt.ylim(-0.5, 0.5)
plt.title('Lowpass FIR Filter')

#plot lowpass iir filter amplitude
plt.subplot(132)
plt.plot(IIR_FilteredSignal, color='b')
plt.ylabel('Amplitude')
plt.xlabel('fs')
plt.ylim(-0.5, 0.5)
plt.title('Lowpass IIR Filter')


#plot louder record amplitude
plt.subplot(132)
plt.plot(record_loud, color='r')
plt.ylabel('Amplitude')
plt.xlabel('fs')
plt.ylim(-0.5, 0.5)
plt.title('Louder record')

#plot quieter record amplitude
plt.subplot(133)
plt.plot(record_quiet, color='r')
plt.ylabel('Amplitude')
plt.xlabel('fs')
plt.ylim(-0.5, 0.5)
plt.title('Quieter record')
'''
'''
#voice changed amplitude
plt.subplot(132)
plt.plot((voice_fir_filtered), color='b')
plt.ylabel('Amplitude')
plt.xlabel('fs')
plt.ylim(-0.5, 0.5)
plt.title('Voice changed')
'''
play_record(voice_record, fs)
play_record(voice_fir_filtered, fs)

plt.show()
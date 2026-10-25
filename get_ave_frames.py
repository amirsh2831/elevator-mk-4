import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

samples = "keyword-samples"
frame_size = 1024

average_frames = []

def extract(path):
    sample_rate, data = wav.read(path)

    if (len(data.shape) == 2):
        data = np.mean(data, axis=1)

    num_frames = (len(data) - frame_size) // (frame_size // 2) + 1
    vad_energies = []

    # ^ apply VAD
    for i in range(num_frames):
        start = i * (frame_size // 2)
        end = start + frame_size
        frame = data[start:end]
        enrg = np.sum(frame ** 2)
        vad_energies.append(enrg)

    nrom_enrg = vad_energies / max(vad_energies)
    w_start = 0
    w_end = 0

    for ws in range(num_frames):
        if (nrom_enrg[ws] >= 0.05):
            w_start = ws
            break

    for we in range(num_frames - 1, -1, -1):
        if (nrom_enrg[we] >= 0.05):
            w_end = we
            break
    
    # ^ get VAD length
    length = w_end - w_start
    vad_value_start = w_start * frame_size // 2
    vad_value_stop = w_end * frame_size // 2
    print(path, length)
    return length

wav_files = [f for f in os.listdir(samples) if f.endswith('.wav')]
wav_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  

for filename in wav_files:
    path = os.path.join(samples, filename)
    average_frames.append(extract(path))

ave = np.mean(average_frames)
print("-" * 100)
print ("Final average framse: ", ave)

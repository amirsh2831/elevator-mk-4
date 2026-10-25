import numpy as np
import csv
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

train_data = "samples"
frame_size = 1024
target_frames = 32

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



    # * feature extraction
    energies = np.zeros(length)
    averages = np.zeros(length)
    zcr = np.zeros(length)

    for n in range(length):
        start = vad_value_start +  n * (frame_size // 2)
        end = start + frame_size 
        frame = data[start:end]

        energy = np.sum(frame ** 2)
        avg = np.mean(np.abs(frame))
        z = 0

        for a in range(1, len(frame)):
            if (frame[a] * frame[a-1] < 0): 
                z += 1
            elif (frame[a] * frame[a-1] == 0):
                if (frame[a] * frame[a-2] < 0):
                    z += 1
        
        energies[n]=energy
        averages[n] = avg
        zcr[n] = z

    # ^ normalize values
    # norm_energies = energies / max(energies)
    # norm_averages = averages / max(averages)
    # norm_zcr = zcr / max(zcr)

    # features = np.array([norm_energies, norm_averages, norm_zcr])
    features = np.array([energies, averages, zcr])
    return features

def resize(features, target):
    # ^ get original and feature counts
    original , feats = features.shape
    resized = np.zeros((target, feats))

    scale = (original - 1 ) / (target - 1)

    for i in range(target - 1):
        index = i * scale
        j = int(index)
        wx = index - j
        if j + 1 < original:
            resized[i] = features[j] * (1 - wx) + features[j+1] * wx
        else : 
            resized[i] = features[j]
    resized[-1] = features[-1]
    return resized


# ^ main app
all_features = []

wav_files = [f for f in os.listdir(train_data) if f.endswith('.wav')]
wav_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # numerical sort by filename number

for filename in wav_files:
    if filename.endswith('.wav'):
        path = os.path.join(train_data, filename)
        feat = extract(path)
        all_features.append(feat)

X = []
y = []

resized_features = []
for i, feat in enumerate(all_features):
    resized = resize(feat, target_frames)
    energy_list = resized[:, 0].tolist()
    avg_list = resized[:, 1].tolist()
    zcr_list = resized[:, 2].tolist()
    combined = np.concatenate([energy_list, avg_list, zcr_list])
    digit_label = i // 10
    X.append(combined)
    y.append(digit_label)

X = np.array(X)
y = np.array(y)

np.save('features.npy', X)
np.save('labels.npy', y)

print("\n=== DONE ===")
print(f"Features shape: {X.shape}")  # Should be (100, 84)
print(f"Labels shape: {y.shape}")
print("Saved features.npy and labels.npy")
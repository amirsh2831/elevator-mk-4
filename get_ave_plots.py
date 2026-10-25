import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

frame_size = 1024
samples = "samples"  

def extract(path):
    sample_rate, data = wav.read(path)
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    num_frames = (len(data) - frame_size) // (frame_size // 2) + 1

    energies = []
    for i in range(num_frames):
        start = i * (frame_size // 2)
        end = start + frame_size

        energy = np.sum(data[start:end] ** 2)
        energies.append(energy)
    
    normalized_energies = energies / max(energies)

    w_start = 0
    w_end = 0


    for ws in range(num_frames):
        if (normalized_energies[ws] >= 0.05):
            w_start = ws
            break
    
    for we in range(num_frames - 1, 0, -1):
        if (normalized_energies[we] >= 0.05):
            w_end = we
            break

    vad_w_stat = w_start * frame_size // 2
    vad_w_end = w_end * frame_size // 2

    return data[vad_w_stat:vad_w_end]

wav_files = [f for f in os.listdir(samples) if f.endswith('.wav')]
wav_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  

all_features = []
filenames = []

for filename in wav_files:
    path = os.path.join(samples, filename)
    feat = extract(path)
    all_features.append(feat)
    filenames.append(filename)


def plot_features(all_features):
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    print(all_features[15])
    averqages = all_features[90:100]
    file = filenames[90:100]
    for i in range(10):
        average = averqages[i]
        x_axis = np.arange(len(average))  
        
        axes[i ].plot(x_axis, average, color='tab:blue', linewidth=1.5)
        axes[i].set_title(f'Digit {file[i]}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('framse')
        axes[i].set_ylabel('average')
        axes[i].grid(alpha=0.3)
   
    plt.tight_layout()
    plt.show()

plot_features(all_features)
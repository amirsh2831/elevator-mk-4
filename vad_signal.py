import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

sample_rate, data = wav.read('keyword-samples/2.wav')
frame_size = 1024
energies = []

if (len(data.shape) == 2):
    data = np.mean(data, axis=1)

num_frames = (len(data) - frame_size) // (frame_size // 2) + 1
print("num_frames: ", num_frames)

for i in range(num_frames):
    start = i * (frame_size // 2)
    end = start + frame_size

    energy = np.sum(data[start:end] ** 2)
    energies.append(energy)

print("single energy)", energies[64])
print("max energy)", max(energies))

normalized_energies = energies / max(energies)

w_start = 0
w_end = 0

for ws in range(num_frames): 
    if (normalized_energies[ws] > 0.03): 
        w_start = ws
        break
        
for we in range(num_frames - 1, 0, -1): 
    if (normalized_energies[we] > 0.03): 
        w_end = we
        break

print("w_start:", w_start)
print("w_end:", w_end)

w_start *=  frame_size // 2
w_end *=  frame_size // 2

print("w_start:", w_start)
print("w_end:", w_end)

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

axs[0].plot(np.arange(len(data)), data,  color='gray'), 
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

axs[1].plot(np.arange(num_frames), normalized_energies, color='blue')
axs[1].set_ylabel("Normalized Energy")
axs[1].grid(True)

axs[2].plot(np.arange(w_end - w_start), data[w_start: w_end], color='red')
axs[2].set_ylabel("Amplitude")
axs[2].grid(True)

plt.show()
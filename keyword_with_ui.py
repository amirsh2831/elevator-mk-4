import numpy as np
import scipy.io.wavfile as wav
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog, messagebox

frame_size = 1024
target_frames = 79
vad_threshold = 0.05

def extract(path):
    sample_rate, data = wav.read(path)
    if (len(data.shape) == 2):
        data = np.mean(data, axis=1)
    num_frames = (len(data) - frame_size) // (frame_size // 2) + 1
    vad_energies = []
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
        if (nrom_enrg[ws] >= vad_threshold):
            w_start = ws
            break
    for we in range(num_frames - 1, -1, -1):
        if (nrom_enrg[we] >= vad_threshold):
            w_end = we
            break
    length = w_end - w_start

    print(length, w_start, w_end)

    vad_value_start = w_start * frame_size // 2
    energies = np.zeros(length)
    averages = np.zeros(length)
    zcr = np.zeros(length)
    for n in range(length):
        start = vad_value_start + n * (frame_size // 2)
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
        energies[n] = energy
        averages[n] = avg
        zcr[n] = z

    if length > 0:
        max_energy = np.max(energies) if np.max(energies) > 0 else 1.0
        max_avg = np.max(averages) if np.max(averages) > 0 else 1.0
        max_zcr = np.max(zcr) if np.max(zcr) > 0 else 1.0
        
        energies = energies / max_energy
        averages = averages / max_avg
        zcr = zcr / max_zcr

    features = np.array([energies, averages, zcr])
    return features

def resize(features, target):
    original, feats = features.shape
    resized = np.zeros((target, feats))
    scale = (original - 1) / (target - 1) if original > 1 else 0
    for i in range(target - 1):
        index = i * scale
        j = int(index)
        wx = index - j
        if j + 1 < original:
            resized[i] = features[j] * (1 - wx) + features[j+1] * wx
        else:
            resized[i] = features[j]
    resized[-1] = features[-1] if original > 0 else 0
    return resized

# MLP Model
class DigitMLP(nn.Module):
    def __init__(self, input_size=237, hidden_size=128, num_classes=2):
        super(DigitMLP, self).__init__()
        self.network = nn.Sequential(
            # * nn.linear -> set random weights and bisases to 0
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            # * instead dif sigmound
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# GUI App Class
class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spoken Digit Recognition GUI")

        self.samples_dir = None
        self.model = None
        self.scaler = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # controls
        self.lr_label = tk.Label(root, text="Learning Rate:")
        self.lr_label.pack()
        self.lr_entry = tk.Entry(root)
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.pack()

        self.epochs_label = tk.Label(root, text="Epochs:")
        self.epochs_label.pack()
        self.epochs_entry = tk.Entry(root)
        self.epochs_entry.insert(0, "200")
        self.epochs_entry.pack()

        self.batch_size_label = tk.Label(root, text="Batch Size:")
        self.batch_size_label.pack()
        self.batch_size_entry = tk.Entry(root)
        self.batch_size_entry.insert(0, "8")
        self.batch_size_entry.pack()

        # buttons
        self.select_folder_btn = tk.Button(root, text="Select Samples Folder", command=self.select_folder)
        self.select_folder_btn.pack()

        self.load_data_btn = tk.Button(root, text="Load and Split Data", command=self.load_data)
        self.load_data_btn.pack()

        self.train_btn = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_btn.pack()

        self.test_file_btn = tk.Button(root, text="Select WAV File to Test", command=self.test_file)
        self.test_file_btn.pack()

        self.export_weights_btn = tk.Button(root, text="Export Weights to TXT", command=self.export_weights)
        self.export_weights_btn.pack()

        # plot 
        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.suptitle("Training Progress")
        self.axs[0].set_title("Loss")
        self.axs[1].set_title("Accuracy")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.train_losses = []
        self.train_accuracies = []

    def select_folder(self):
        self.samples_dir = filedialog.askdirectory(title="Select Samples Folder")
        if self.samples_dir:
            messagebox.showinfo("Selected", f"Folder selected: {self.samples_dir}")

    def load_data(self):
        if not self.samples_dir:
            messagebox.showerror("Error", "Select samples folder first!")
            return

        all_features = []
        wav_files = sorted([f for f in os.listdir(self.samples_dir) if f.endswith('.wav')], key=lambda x: int(os.path.splitext(x)[0]))

        for filename in wav_files:
            path = os.path.join(self.samples_dir, filename)
            feat = extract(path)
            feat = feat.T
            resized = resize(feat, target_frames)
            combined = resized.flatten()
            print(f"{filename}: shape = {combined.shape}")
            all_features.append(combined)

        X = np.array(all_features)
        y = np.array([1 if i < 10 else 0 for i in range(len(all_features))])

        #  7 train + 3 test per digit
        train_indices = list(range(0, 7)) + list(range(10, 17))  # 7 keyword + 7 non-keyword
        test_indices = list(range(7, 10)) + list(range(17, 20))   # 3 keyword + 3 non-keyword
        # train_indices = []
        # test_indices = []
        # for digit in range(10):
        #     digit_indices = np.where(y == digit)[0]
        #     train_indices.extend(digit_indices[:7])
        #     test_indices.extend(digit_indices[7:10])

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        # Scale
        # * standart data with mean = 0 and deviation = 1
        # * for more standard data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_train = X_train_scaled
        self.y_train = y_train
        self.X_test = X_test_scaled
        self.y_test = y_test

        messagebox.showinfo("Data Loaded", f"Train: {len(self.y_train)} samples, Test: {len(self.y_test)} samples")

    def train_model(self):
        if self.X_train is None:
            messagebox.showerror("Error", "Load data first!")
            return

        lr = float(self.lr_entry.get())
        epochs = int(self.epochs_entry.get())
        batch_size = int(self.batch_size_entry.get())

        self.model = DigitMLP()
        # * instead of softmax on (output)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # DataLoader
        train_dataset = TensorDataset(torch.FloatTensor(self.X_train), torch.LongTensor(self.y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.train_losses = []
        self.train_accuracies = []

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                # * forward pass
                outputs = self.model(inputs)
                # * for backward propagation and loss calculation
                loss = criterion(outputs, labels)
                # * backward propagation
                loss.backward()
                # * instead of manual updates weights -= learning_rate * gradient
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)

            # Update plot every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.update_plot()
                self.root.update()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.FloatTensor(self.X_test))
            _, predicted = torch.max(outputs, 1)
            test_acc = accuracy_score(self.y_test, predicted.numpy())

        messagebox.showinfo("Training Complete", f"Test Accuracy: {test_acc*100:.2f}%")

    def update_plot(self):
        self.axs[0].cla()
        self.axs[0].plot(self.train_losses)
        self.axs[0].set_title("Loss")
        self.axs[0].grid(alpha=0.3)

        self.axs[1].cla()
        self.axs[1].plot(self.train_accuracies)
        self.axs[1].set_title("Accuracy")
        self.axs[1].grid(alpha=0.5)

        self.canvas.draw()

    def test_file(self):
        if self.model is None:
            messagebox.showerror("Error", "Train model first!")
            return

        file_path = filedialog.askopenfilename(title="Select WAV File", filetypes=[("WAV files", "*.wav")])
        if file_path:
            feat = extract(file_path)
            feat = feat.T
            resized = resize(feat, target_frames)
            combined = resized.flatten()
            for b in range(len(combined)):
                print(combined[b], ",")
            scaled = self.scaler.transform([combined])
            input_tensor = torch.FloatTensor(scaled)
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                _, predicted = torch.max(output, 1)
            result = "KEYWORD DETECTED" if predicted.item() == 1 else "NOT KEYWORD"
            messagebox.showinfo("Prediction", f"Result: {result}")
            # digit = predicted.item() + 1  # 0->1, 1->2, etc.
            # messagebox.showinfo("Prediction", f"Predicted Digit: {digit}")

    def export_weights(self):
        if self.model is None:
            messagebox.showerror("Error", "Train the model first!")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            title="Save Weights as C Arrays"
        )
        if not save_path:
            return
        with open(save_path, 'w') as f:
            f.write("// Weights and Biases for ESP32\n")
            f.write("// Input: 237 - Hidden1: 128 - Hidden2: 64 - Output: 2\n")
            f.write("// Use float arrays in C\n\n")
            for name, param in self.model.named_parameters():
                arr = param.data.cpu().numpy()
                flat = arr.flatten()
                c_name = name.replace('.', '_')
                f.write(f"const float {c_name}[{flat.size}] = {{\n")
                for i in range(0, len(flat), 8):
                    chunk = flat[i:i+8]
                    line = ", ".join([f"{v:.6f}f" for v in chunk])
                    f.write("  " + line)
                    if i + 8 < len(flat):
                        f.write(",")
                    f.write("\n")
                f.write("};\n\n")
            
            # FIX: Use scaler.scale_ directly (it's already 1/std)
            f.write("// StandardScaler parameters\n")
            mean = self.scaler.mean_
            scale = self.scaler.scale_  # ← CHANGED: Use scale_ directly
            
            f.write(f"const float scaler_mean[{len(mean)}] = {{\n")
            for i in range(0, len(mean), 8):
                chunk = mean[i:i+8]
                line = ", ".join([f"{v:.6f}f" for v in chunk])
                f.write("  " + line)
                if i + 8 < len(mean):
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")
            
            f.write(f"const float scaler_scale[{len(scale)}] = {{\n")
            for i in range(0, len(scale), 8):
                chunk = scale[i:i+8]
                line = ", ".join([f"{v:.6f}f" for v in chunk])
                f.write("  " + line)
                if i + 8 < len(scale):
                    f.write(",")
                f.write("\n")
            f.write("};\n")
        messagebox.showinfo("Success", f"Weights exported to:\n{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    root.mainloop()
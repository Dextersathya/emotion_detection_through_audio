
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle

!kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio

!unzip ravdess-emotional-speech-audio.zip

audio_files=glob('/content/audio_speech_actors_01-24/*/*.wav')

len(glob('/content/audio_speech_actors_01-24/*/*.wav'))

ipd.Audio(audio_files[0])

y,sr=librosa.load(audio_files[0])

y

pd.Series(y).plot(figsize=(10,5),lw=1)
plt.show()

y_trimed,_=librosa.effects.trim(y,top_db=20)

pd.Series(y_trimed).plot(figsize=(10,5),lw=1)
plt.show()

pd.Series(y).plot(figsize=(10,5),lw=1)
plt.show()

D=librosa.stft(y)
s_db=librosa.amplitude_to_db(np.abs(D),ref=np.max)
s_db.shape

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
img=librosa.display.specshow(s_db,x_axis='time',y_axis='log',ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.f dB")

"""# Mel Spectogram"""

s=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128 *2,)
s_db_mel=librosa.amplitude_to_db(s,ref=np.max)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
img=librosa.display.specshow(s_db_mel,x_axis='time',y_axis='log',ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.f dB")

Ravdess = "/content/audio_speech_actors_01-24"
Ravdess

import os
import glob
import pandas as pd

# Define the root directory for the RAVDESS dataset
ravdess_dir = Ravdess # Replace with your actual directory if different

# Use glob to find all files with a .wav extension in all subdirectories
file_list = glob.glob(os.path.join(ravdess_dir, '**/*.wav'), recursive=True)

file_emotion = []
file_path = []

for file in file_list:
    # Split the file name to extract the emotion
    part = os.path.basename(file).split('.')[0].split('-')
    # The third part in each file represents the emotion associated with that file
    file_emotion.append(int(part[2]))
    # Store the full path
    file_path.append(file)

# Create a dataframe for the emotions of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# Create a dataframe for the paths of files
path_df = pd.DataFrame(file_path, columns=['Path'])

# Concatenate the dataframes along the columns
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# Replace emotion numbers with actual emotion labels
Ravdess_df.Emotions.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)

# Display the first few rows of the dataframe
Ravdess_df.head()

Ravdess_df

csv_filename = 'Ravdess_emotion_data.csv'

Ravdess_df.to_csv(csv_filename, index=False)

print(f"DataFrame has been saved as {csv_filename}")

emotion_mapping={
    'neutral':0,
    'calm':1,
    'happy':2,
    'sad':3,
    'angry':4,
    'fear':5,
    'disgust':6,
    'surprise':7

}

Ravdess_df['Label'] = Ravdess_df['Emotions'].map(emotion_mapping)

Ravdess_df

import matplotlib.pyplot as plt
import seaborn as sns

emotion_colors = {
    'neutral': '#1f77b4',   # Blue
    'calm': '#ff7f0e',      # Orange
    'happy': '#2ca02c',     # Green
    'sad': '#d62728',       # Red
    'angry': '#9467bd',     # Purple
    'fear': '#8c564b',      # Brown
    'disgust': '#e377c2',   # Pink
    'surprise': '#7f7f7f'   # Grey
}

emotion_palette = [emotion_colors[emotion] for emotion in Ravdess_df['Emotions'].unique()]
plt.figure(figsize=(10, 6))
plt.title('Count of Emotions', size=16)
sns.countplot(x='Emotions', data=Ravdess_df, palette=emotion_palette)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
plt.xticks(rotation=30, size=10)  # Optional: Rotate x-axis labels for better visibility
plt.show()

def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def create_spectrogram(data, sr, e):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

emotion='happy'
path = np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[5]
data, sampling_rate = librosa.load(path)
create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)

emotion='sad'
path = np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]
data, sampling_rate = librosa.load(path)
create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)
ipd.Audio(path)

"""# **DATA AGUMENTATION**"""

def noise(data):
    noise_amp = 0.01 *np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data):
    return librosa.effects.time_stretch(data, rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, n_steps=2):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=0.7)



path = np.array(Ravdess_df.Path)[2]
data, sample_rate = librosa.load(path)

x = noise(data)
plt.figure(figsize=(14,4))
librosa.display.waveshow(y=x, sr=sample_rate)
ipd.Audio(x, rate=sample_rate)

plt.figure(figsize=(14,4))
librosa.display.waveshow(y=data, sr=sample_rate)
ipd.Audio(path)

"""# **FEATURE EXTRACTION**"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sample_rate = 16000
n_mels = 128
hop_length = 635
f_max = 8000
n_fft = 800
win_length = 800

output_dir = '/content/mel_spectrograms'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def extract_mel_spectrogram(file_path, sample_rate, output_file, n_mels, hop_length,fmax,labels):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)

        mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=fmax)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        output_file = os.path.join(output_dir, f"{labels}_{os.path.basename(file_path).split('.')[0]}.png")
        plt.figure(figsize=(10,4))
        librosa.display.specshow(mel_spect_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.axis('off')
#         plt.colorbar(format='%+2.0f dB')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing file {file_path}:{e}")

for index, row in Ravdess_df.iterrows():
    file_path = row['Path']
    emotion = row['Label']
    extract_mel_spectrogram(file_path,sample_rate,output_dir,n_mels,hop_length,f_max,emotion)

print(f"Mel-Spectrograms saved to {output_dir}")

mels_dir = '/content/mel_spectrograms'

if os.path.exists(mels_dir):
    num_files = len(os.listdir(mels_dir))
else:
    num_files = 0

num_files

"""# **LLFs Features**"""

from sklearn.preprocessing import StandardScaler
from skimage.transform import resize

def extract_llf_features(audio_data, sr, n_fft, win_length, hop_length):

    rms = librosa.feature.rms(y=audio_data, frame_length=win_length, hop_length=hop_length, center=False)
    chroma = librosa.feature.chroma_stft(n_chroma=17,y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    poly_features = librosa.feature.poly_features(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    zcr = librosa.feature.zero_crossing_rate(y=audio_data, frame_length=win_length, hop_length=hop_length, center=False)
    feats = np.vstack((chroma, #12
                spectral_contrast, #7
                spectral_centroid, #1
                spectral_bandwidth, #1
                spectral_flatness, #1
                spectral_rolloff, #1
                poly_features, #2
                rms, #1
                zcr #1
                ))

    scaler = StandardScaler()
    feats = scaler.fit_transform(feats.T).T
    feats = librosa.power_to_db(feats)

    return feats

llfs_output_dir='/content/llfs'
if not os.path.exists(llfs_output_dir):
    os.makedirs(llfs_output_dir)

def save_llf(file_path, sample_rate,  output_dir, labels, n_fft=2048, win_length=1024, hop_length=512):
    try:
        audio_data, _ = librosa.load(file_path, sr=sample_rate)

        if audio_data.size == 0:
            print(f"Warning: Audio data is empty for file {file_path}. Skipping this file.")
            return None

        mfcc_db= extract_llf_features(audio_data,sample_rate, n_fft, win_length, hop_length)


        if mfcc_db.size == 0:
            print(f"Warning: MFCC features are empty for file {file_path}. Skipping this file.")
            return None

        delta_mfcc = librosa.feature.delta(mfcc_db)
        delta2_mfcc = librosa.feature.delta( mfcc_db, order=2)

        combined_features = np.concatenate([mfcc_db, delta_mfcc, delta2_mfcc], axis=1)
        combined_features = resize(combined_features, (32, 128), anti_aliasing=True)




        output_file = os.path.join(output_dir, f"{labels}_{os.path.basename(file_path).split('.')[0]}.png")

        plt.figure(figsize=(10, 4))
        plt.imshow(combined_features, cmap='viridis', aspect='auto', origin='lower')
#         plt.colorbar(format='%+2.0f dB')
        plt.xticks([])   # Ẩn nhãn trục X
        plt.yticks([])   # Ẩn nhãn trục Y
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()

        return output_file
    except Exception as e:
        print(f"Error processing file {file_path} for LLF extraction: {e}")

for index, row in Ravdess_df.iterrows():
    file_path = row['Path']
    emotion = row['Label']
    save_llf(file_path, sample_rate, llfs_output_dir,emotion,n_fft, win_length , hop_length)
print(f"LLFs saved to {llfs_output_dir}")

"""# **combining**"""

from PIL import Image

def combine_images_if_same_filename(mel_spectrogram_path, llf_path, output_dir):
    try:
        mel_filename = os.path.basename(mel_spectrogram_path).split('.')[0]
        llf_filename = os.path.basename(llf_path).split('.')[0]

        # Kiểm tra nếu tên file giống nhau
        if mel_filename == llf_filename:
            print(f"Combining {mel_filename} and {llf_filename}")

            # Đọc ảnh Mel-spectrogram và LLF
            mel_img = Image.open(mel_spectrogram_path)
            llf_img = Image.open(llf_path)

            # Chọn kích thước lớn nhất giữa hai ảnh
            new_width = max(mel_img.width, llf_img.width)
            new_height = max(mel_img.height, llf_img.height)

            # Resize cả hai ảnh về cùng kích thước
            mel_img_resized = mel_img.resize((new_width, new_height))
            llf_img_resized = llf_img.resize((new_width, new_height))

            combined_img = np.vstack((np.array(mel_img_resized), np.array(llf_img_resized)))

            # Convert lại thành ảnh
            combined_img = Image.fromarray(combined_img)

            # Tạo thư mục lưu nếu chưa tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Lưu ảnh kết hợp với tên file giống với file Mel hoặc LLF
            output_file = os.path.join(output_dir, f"{mel_filename}.png")
            combined_img.save(output_file)

            # Hiển thị ảnh đã kết hợp
            plt.imshow(combined_img)
            plt.axis('off')
#             plt.show()

            return output_file
        else:
            print(f"Filenames do not match: {mel_filename} and {llf_filename}. Skipping.")
            return None
    except Exception as e:
        print(f"Error combining images: {e}")
        return None

combine_output_dir='/content/combine_image'
if not os.path.exists(combine_output_dir):
    os.makedirs(combine_output_dir)

sample_combine_mel = '/content/mel_spectrograms/0_03-01-01-01-01-01-01.png'
sample_combine_llf = '/content/llfs/0_03-01-01-01-01-01-01.png'
combine_images_if_same_filename(sample_combine_mel, sample_combine_llf , combine_output_dir)

mel_dir = '/content/mel_spectrograms'
llf_dir = '/content/llfs'
combine_output_dir = '/content/combine_image'

if not os.path.exists(combine_output_dir):
    os.makedirs(combine_output_dir)

for mel_file in os.listdir(mel_dir):
    mel_file_path = os.path.join(mel_dir, mel_file)

    if os.path.isfile(mel_file_path):

        mel_filename = mel_file.split('.')[0]
        llf_file_path = os.path.join(llf_dir, f"{mel_filename}.png")


        if os.path.isfile(llf_file_path):

            combine_images_if_same_filename(mel_file_path, llf_file_path, combine_output_dir)
        else:
            print(f"LLF file not found for {mel_filename}. Skipping.")

import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

class EmotionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Đảm bảo hình ảnh có 3 kênh

        # Lấy nhãn từ tên file
        label = int(self.image_files[idx].split('_')[0])  # Nhãn được lưu ở phần đầu tên file

        if self.transform:
            image = self.transform(image)

        # In kích thước hình ảnh và loại
#         print(f"Image size: {image.size} | Label: {label}")  # In kích thước hình ảnh và nhãn
#         print(f"Image tensor shape: {image.shape}")  # In kích thước tensor hình ảnh

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Thay đổi kích thước

    transforms.RandomHorizontalFlip(),  # Lật ngẫu nhiên theo chiều ngang
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Thay đổi độ sáng, độ tương phản, độ bão hòa và sắc độ

    transforms.ToTensor(),           # Chuyển đổi sang tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_dir = '/content/combine_image'
dataset = EmotionDataset(image_dir, transform=transform)

# Tạo DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Kiểm tra dữ liệu
for images, labels in data_loader:
    print(images.shape)
    print(labels)
    break

"""# **Split Dataset**"""

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f'Train dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomResNet50, self).__init__()

        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)

        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Replace the fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomResNet50(num_classes=8).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # Tính toán hàm mất mát
            loss.backward()  # Backward pass
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # Lấy nhãn dự đoán
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Đếm số dự đoán đúng

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')



def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)
test_model(model, test_loader, criterion)
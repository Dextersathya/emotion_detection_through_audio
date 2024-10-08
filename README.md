# **RAVDESS Audio Emotion Analysis**

This project focuses on analyzing and extracting features from the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. It demonstrates data preprocessing, feature extraction, and visualization techniques, and lays the groundwork for building emotion recognition models using audio data.

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Setup Instructions](#setup-instructions)
4. [Feature Extraction and Visualization](#feature-extraction-and-visualization)
5. [Data Augmentation](#data-augmentation)
6. [Mel-Spectrogram Extraction](#mel-spectrogram-extraction)
7. [Low-Level Feature (LLF) Extraction](#low-level-feature-llf-extraction)
8. [Combining Mel-Spectrograms and LLF](#combining-mel-spectrograms-and-llf)
9. [Usage](#usage)
10. [Acknowledgements](#acknowledgements)

## **Project Overview**

The objective of this project is to extract meaningful features from audio files in the RAVDESS dataset, which contains emotional speech from different actors. This project performs:

- Audio loading and visualization
- Data augmentation to generate variations in audio files
- Feature extraction such as Mel-Spectrograms and low-level features (LLFs)
- Visualization of features
- Saving the extracted features in image formats for further processing

## **Dataset**

The project uses the [RAVDESS Emotional Speech Audio dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio). It contains speech data with 8 different emotions: neutral, calm, happy, sad, angry, fear, disgust, and surprise.

### **Dataset Structure**

The dataset is structured with different actors and emotions as subdirectories. Each audio file is named with a specific format that indicates emotion, actor, and other attributes.

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone <repository-url>
cd <repository-directory>
```

### **2. Install Dependencies**

```bash
pip install pandas numpy matplotlib seaborn librosa tqdm torch torchvision scikit-image
```

### **3. Download the Dataset**

Download the RAVDESS dataset using Kaggle and unzip it in the root directory of this project. Ensure that the directory structure looks like this:

```
/content
    ├── audio_speech_actors_01-24
    │   ├── Actor_01
    │   ├── Actor_02
    │   └── ...
```

### **4. Run the Code**

You can execute the code block by block in a Jupyter Notebook or run the entire script if you're using an IDE.

## **Feature Extraction and Visualization**

The code provides utilities to:

- Load and visualize the waveform of audio files using `create_waveplot()`.
- Display Mel-Spectrograms and Spectrograms for different emotions.
- Perform feature extraction such as Mel-Frequency Cepstral Coefficients (MFCCs), chroma features, spectral contrast, and zero-crossing rate.

## **Data Augmentation**

To create variations in the dataset and improve model generalization, the following data augmentation techniques are applied:

- Adding random noise
- Time-stretching
- Shifting
- Pitch modulation

These augmentations help in creating diverse training samples and are implemented using functions like `noise()`, `stretch()`, `shift()`, and `pitch()`.

## **Mel-Spectrogram Extraction**

The project extracts Mel-Spectrograms for each audio file and saves them as PNG images. This step is crucial for visual representation and can be used for training Convolutional Neural Networks (CNNs).

## **Low-Level Feature (LLF) Extraction**

Low-level features such as RMS, chroma, spectral centroid, and bandwidth are extracted for each audio file. These features are scaled, transformed, and saved as images to be used in conjunction with Mel-Spectrograms for better emotion recognition.

## **Combining Mel-Spectrograms and LLF**

The project includes functionality to combine Mel-Spectrogram and LLF images for each audio sample if their filenames match, allowing the creation of a comprehensive feature set.

## **Usage**

1. **Exploring Audio Files**: The initial cells of the script load and display waveforms of audio files.
2. **Generating Mel-Spectrograms**: Run the extraction function for Mel-Spectrogram generation and visualization.
3. **Data Augmentation**: Execute the augmentation functions to generate variations of the original audio files.
4. **Extracting LLF Features**: The code will compute and save LLF features for each file.
5. **Combining Features**: Combine the Mel-Spectrogram and LLF features into a single image for further analysis.

## **Acknowledgements**

- This project uses the RAVDESS dataset, which is a valuable resource for emotion recognition research.
- The project leverages libraries such as `librosa` for audio processing and `matplotlib`/`seaborn` for visualization.

For further inquiries or contributions, feel free to reach out or create a pull request.

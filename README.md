# 🎹 Two-Handed Polyphonic Synth
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00BFFF?style=flat&logo=google&logoColor=white)](https://developers.google.com/mediapipe)

An interactive, browser-based radial synthesizer powered by a dual-machine learning architecture. This application bridges computer vision and audio signal processing to create a completely hands-free musical instrument. 

Users can play complex polyphonic chords using spatial hand gestures, while the system continuously listens to the environment—using acoustic timbre classification to dynamically alter the synthesizer's waveform in real-time.

---

## 📸 Interface Preview
![App Interface](assets/interface_preview.png)

---

## ✨ System Architecture 

This project utilizes a split-inference architecture to ensure zero-latency musical playback while handling heavy audio-classification processing in the background.

### 1. Client-Side Vision Pipeline (Zero-Latency UI)
* **Hand Tracking:** Google's **MediaPipe** captures 21 3D landmarks for both hands at 30+ FPS.
* **Trigonometric Mapping:** Hand vector angles are calculated relative to the wheel center to select notes (Left Wheel, 12 slices) and chord types (Right Wheel, 8 slices).
* **Edge Inference (ONNX):** A custom gesture model exported to `.onnx` runs entirely in the browser using `ONNXRuntime-Web`.
* **Sound Generation:** The **HTML5 Web Audio API** dynamically generates polyphonic waveforms on the fly with instant touchless index-finger tracking.

### 2. Server-Side Audio Pipeline (Timbre Classification)
* **Live Streaming:** `streamlit-webrtc` pipes chunks of microphone audio from the browser to the Python backend.
* **Feature Extraction:** **Librosa** extracts 13 Mel-frequency cepstral coefficients (MFCCs) from 1-second audio buffers.
* **Deep Learning (Bi-LSTM):** A **Bidirectional LSTM** (`timbre_bilstm.h5`) built with TensorFlow/Keras classifies the acoustic texture.
* **State Management:** The backend dynamically updates the Streamlit state, swapping the browser's audio oscillators from Sine waves to Sawtooth waves based on the environment.

---

## 🎮 How to Play

1. **Launch the App:** Run `streamlit run app.py` and click the **Initialize Synth** button to grant camera and microphone permissions.
2. **Aim the Left Hand (Notes):** Point your left index finger over the left radial menu to select the root note (C through B).
3. **Aim the Right Hand (Chords):** Point your right index finger over the right radial menu to select the chord type (Major, Minor, 7th, Diminished, etc.).
4. **Change the Waveform:** Make soft, sustained sounds into your microphone to trigger a **Sine** wave, or sharp, percussive sounds to shift to a **Sawtooth** wave via backend Bi-LSTM classification.

---

## 🚀 Installation & Setup

### Prerequisites
Make sure you have Python 3.9+ installed and a working webcam/microphone.

### 1. Clone the Repository
```bash
git clone [https://github.com/Viper2911/Two-Handed-Polyphonic-Synth.git](https://github.com/Viper2911/Two-Handed-Polyphonic-Synth.git)
cd Two-Handed-Polyphonic-Synth
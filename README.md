# 🎹 Two-Handed Polyphonic Synth
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00BFFF?style=flat&logo=google&logoColor=white)](https://developers.google.com/mediapipe)

An interactive, browser-based radial synthesizer powered by a dual-machine learning architecture. This application bridges computer vision and audio signal processing to create a completely hands-free musical instrument. 

Users can play complex polyphonic chords using spatial hand gestures, while the system continuously listens to the environment—using acoustic timbre classification to dynamically alter the synthesizer's waveform in real-time.

---

## 📸 Interface Preview
<!-- 💡 Pro-Tip: Record a 5-second GIF of you playing the synth and upload it to your repo, then replace the image path below with your GIF! -->
![App Interface](https://via.placeholder.com/800x400.png?text=Add+a+Screenshot+or+GIF+of+the+Radial+UI+Here)

---

## ✨ System Architecture 

This project utilizes a split-inference architecture to ensure zero-latency musical playback while handling heavy audio-classification processing in the background.

### 1. Client-Side Vision Pipeline (Zero-Latency UI)
* **Hand Tracking:** Google's **MediaPipe** captures 21 3D landmarks for both hands at 30+ FPS.
* **Trigonometric Mapping:** Hand vector angles are calculated relative to the wrist to select notes (Left Hand, 12 slices) and chord types (Right Hand, 8 slices).
* **Edge Inference (ONNX):** A custom **Random Forest** model, trained in Python and exported to `.onnx`, runs entirely in the browser using `ONNXRuntime-Web`. It detects a "Pinch" gesture to trigger the audio playback without making server calls.
* **Sound Generation:** The **HTML5 Web Audio API** dynamically generates polyphonic waveforms on the fly.

### 2. Server-Side Audio Pipeline (Timbre Classification)
* **Live Streaming:** `streamlit-webrtc` pipes chunks of microphone audio from the browser to the Python backend.
* **Feature Extraction:** **Librosa** extracts 13 Mel-frequency cepstral coefficients (MFCCs) from 1-second audio buffers.
* **Deep Learning (Bi-LSTM):** A **Bidirectional LSTM** built with TensorFlow/Keras classifies the acoustic texture (e.g., a breathy hum vs. a percussive beatbox).
* **State Management:** The backend dynamically updates the Streamlit state, swapping the browser's audio oscillators from Sine waves to Sawtooth waves based on the environment.

---

## 🎮 How to Play

1. **Aim the Left Hand (Notes):** Point your left index finger like the hand of a clock. The angle selects the root note (A through G#) on the left radial menu.
2. **Aim the Right Hand (Chords):** Point your right index finger to select the chord progression (Major, Minor, Maj7, Diminished, etc.) on the right radial menu.
3. **Trigger the Sound:** Bring your left thumb and index finger together in a **Pinch** gesture to strike the chord. Open your hand to stop playing.
4. **Change the Waveform:** Hum smoothly into the microphone to switch to a soft **Sine** wave. Make harsh, percussive sounds (like beatboxing) to switch to a gritty **Sawtooth** wave.

---

## 🚀 Installation & Setup

### Prerequisites
Make sure you have Python 3.9+ installed and a working webcam/microphone.

### 1. Clone the Repository
```bash
git clone [https://github.com/Viper2911/Two-Handed-Polyphonic-Synth.git](https://github.com/Viper2911/Two-Handed-Polyphonic-Synth.git)
cd Two-Handed-Polyphonic-Synth
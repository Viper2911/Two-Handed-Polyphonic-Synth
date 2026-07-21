"""
Project: AI Radial Synthesizer - Full Dual-ML Integration
Author: Aadil Hasan
Registration Number: 5000
"""

import streamlit as st
import base64
import os
import numpy as np
import librosa
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide", page_title="AI Radial Synth")

st.markdown("""
    <style>
        #MainMenu {visibility:hidden;}
        header {visibility:hidden;}
        footer {visibility:hidden;}
        .block-container {padding:0;max-width:100%;}
    </style>
""", unsafe_allow_html=True)

if 'wave_type' not in st.session_state:
    st.session_state.wave_type = "sine"

@st.cache_resource
def load_audio_model():
    model_path = 'timbre_bilstm.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

bilstm_model = load_audio_model()

class TimbreAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = np.array([])
        self.sample_rate = 16000
        self.model = bilstm_model
        
    def recv(self, frame):
        audio_chunk = frame.to_ndarray()
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
        if len(self.audio_buffer) >= self.sample_rate:
            audio_data = self.audio_buffer[:self.sample_rate].astype(np.float32)
            if self.model is not None:
                mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
                mfccs_processed = np.expand_dims(mfccs.T, axis=0)
                
                prediction = self.model.predict(mfccs_processed, verbose=0)
                class_id = np.argmax(prediction)

                if class_id == 0:
                    st.session_state.wave_type = "sine"
                else:
                    st.session_state.wave_type = "sawtooth"
            self.audio_buffer = np.array([])
        return frame

with st.container():
    st.markdown("""
        <div style='position:absolute; top:20px; width:100%; display:flex; justify-content:center; z-index:100;'>
            <div style='background:rgba(20,20,20,0.8); color:#00ffcc; padding:8px 20px; border-radius:20px; border:1px solid #00ffcc; font-family:sans-serif; font-size:14px; font-weight:bold; box-shadow:0 0 10px rgba(0,255,204,0.3); backdrop-filter:blur(5px);'>
                🎙️ TIMBRE CLASSIFICATION ACTIVE
            </div>
        </div>
    """, unsafe_allow_html=True)
    webrtc_streamer(
        key="timbre_classifier",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=TimbreAudioProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

model_path_onnx = "gesture_model.onnx"
onnx_base64 = ""
if os.path.exists(model_path_onnx):
    with open(model_path_onnx, "rb") as f:
        onnx_base64 = base64.b64encode(f.read()).decode('utf-8')
else:
    st.error(f"Vision Model not found. Ensure '{model_path_onnx}' is present in the directory.")

html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <script type="module">
        import {{ HandLandmarker, FilesetResolver }} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs";

        const video = document.getElementById('webcam');
        const canvas = document.getElementById('output_canvas');
        const ctx = canvas.getContext('2d');
        const overlayBtn = document.getElementById('overlayBtn');

        let handLandmarker, onnxSession, audioCtx, masterGain;
        let oscillators = [];
        let noteGains = [];
        const NUM_VOICES = 4;
        
        let currentWaveform = "{st.session_state.wave_type}";

        const noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        const baseFreqs = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88];
        const chordTypes = [
            {{ name: "maj", intervals: [0, 4, 7] }},
            {{ name: "maj7", intervals: [0, 4, 7, 11] }},
            {{ name: "7", intervals: [0, 4, 7, 10] }},
            {{ name: "sus4", intervals: [0, 5, 7] }},
            {{ name: "m", intervals: [0, 3, 7] }},
            {{ name: "m7", intervals: [0, 3, 7, 10] }},
            {{ name: "dim", intervals: [0, 3, 6] }},
            {{ name: "aug", intervals: [0, 4, 8] }}
        ];

        function initAudio() {{
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AudioContext();
            masterGain = audioCtx.createGain();
            masterGain.gain.value = 0.3; 
            masterGain.connect(audioCtx.destination);
            
            for(let i=0; i<NUM_VOICES; i++) {{
                let osc = audioCtx.createOscillator();
                let gain = audioCtx.createGain();
                osc.type = currentWaveform; 
                gain.gain.value = 0;
                osc.connect(gain);
                gain.connect(masterGain);
                osc.start();
                oscillators.push(osc);
                noteGains.push(gain);
            }}
        }}

        function playChord(rootIndex, chordIndex, shouldPlay) {{
            if (!audioCtx) return;
            const rootFreq = baseFreqs[rootIndex];
            const chord = chordTypes[chordIndex];
            
            if (shouldPlay) {{
                for (let i = 0; i < NUM_VOICES; i++) {{
                    if (i < chord.intervals.length) {{
                        const freq = rootFreq * Math.pow(2, chord.intervals[i] / 12);
                        oscillators[i].frequency.setTargetAtTime(freq, audioCtx.currentTime, 0.02);
                        noteGains[i].gain.setTargetAtTime(1.0 / chord.intervals.length, audioCtx.currentTime, 0.05);
                    }} else {{
                        noteGains[i].gain.setTargetAtTime(0, audioCtx.currentTime, 0.05);
                    }}
                }}
            }} else {{
                for (let i = 0; i < NUM_VOICES; i++) {{
                    noteGains[i].gain.setTargetAtTime(0, audioCtx.currentTime, 0.05);
                }}
            }}
        }}

        function getAngle(wrist, indexTip) {{
            let dx = wrist.x - indexTip.x; 
            let dy = wrist.y - indexTip.y; 
            let angle = Math.atan2(dx, dy) * (180 / Math.PI);
            if (angle < 0) angle += 360;
            return angle;
        }}

        function drawFixedRadialMenu(cx, cy, radius, numSlices, labels, activeIndex, centerText, isPlaying) {{
            ctx.fillStyle = "rgba(15, 15, 15, 0.7)";
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
            ctx.fill();

            const sliceAngle = (2 * Math.PI) / numSlices;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.font = "bold 15px Arial";

            for (let i = 0; i < numSlices; i++) {{
                const startAngle = i * sliceAngle - Math.PI / 2 - sliceAngle / 2;
                const endAngle = startAngle + sliceAngle;

                if (i === activeIndex) {{
                    ctx.fillStyle = isPlaying ? "rgba(0, 255, 128, 0.5)" : "rgba(0, 255, 204, 0.4)";
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.arc(cx, cy, radius, startAngle, endAngle);
                    ctx.fill();
                    
                    ctx.shadowBlur = 15;
                    ctx.shadowColor = isPlaying ? "lime" : "#00ffcc";
                }}

                ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx + Math.cos(startAngle) * radius, cy + Math.sin(startAngle) * radius);
                ctx.stroke();
                ctx.shadowBlur = 0;

                const textAngle = startAngle + sliceAngle / 2;
                const textX = cx + Math.cos(textAngle) * radius * 0.75;
                const textY = cy + Math.sin(textAngle) * radius * 0.75;
                ctx.fillStyle = i === activeIndex ? "#fff" : "rgba(255, 255, 255, 0.6)";
                ctx.fillText(labels[i], textX, textY);
            }}

            const innerRadius = radius * 0.35;
            ctx.fillStyle = isPlaying ? "rgba(0, 255, 128, 0.2)" : "rgba(20, 20, 20, 0.9)";
            ctx.beginPath();
            ctx.arc(cx, cy, innerRadius, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = isPlaying ? "#0f0" : "#fff";
            ctx.font = "bold 18px Arial";
            ctx.fillText(centerText, cx, cy);
            
            ctx.strokeStyle = "rgba(255,255,255,0.2)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(cx, cy, innerRadius, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
            ctx.stroke();
        }}

        overlayBtn.addEventListener('click', async () => {{
            try {{
                overlayBtn.innerText = "1/4: Initializing Audio...";
                initAudio();
                if(audioCtx.state === 'suspended') await audioCtx.resume();

                overlayBtn.innerText = "2/4: Loading ONNX Model...";
                const b64 = "{onnx_base64}";
                if (b64.length > 10) {{
                    const modelBytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                    onnxSession = await ort.InferenceSession.create(modelBytes);
                }}

                overlayBtn.innerText = "3/4: Loading MediaPipe...";
                const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
                handLandmarker = await HandLandmarker.createFromOptions(vision, {{
                    baseOptions: {{
                        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                        delegate: "GPU"
                    }},
                    runningMode: "VIDEO",
                    numHands: 2
                }});

                overlayBtn.innerText = "4/4: Requesting Camera...";
                const stream = await navigator.mediaDevices.getUserMedia({{ video: {{ width: 1280, height: 720 }} }});
                video.srcObject = stream;
                
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
                
                video.addEventListener('loadeddata', predictWebcam);
                overlayBtn.style.display = "none";
                
            }} catch (error) {{
                overlayBtn.style.backgroundColor = "rgba(200, 0, 0, 0.9)";
                overlayBtn.innerText = "ERROR: " + error.message;
                console.error(error);
            }}
        }});

        let lastVideoTime = -1;
        let currentRootIndex = 0; 
        let currentChordIndex = 0;
        let lastRootIndex = -1;
        let lastChordIndex = -1;
        let lastShouldPlay = false;

        async function predictWebcam() {{
            let startTimeMs = performance.now();
            
            if (lastVideoTime !== video.currentTime) {{
                lastVideoTime = video.currentTime;
                const results = handLandmarker.detectForVideo(video, startTimeMs);
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                let shouldPlay = false;

                const wheelRadius = Math.min(canvas.width, canvas.height) * 0.3;
                const leftWheelX = canvas.width * 0.25;
                const rightWheelX = canvas.width * 0.75;
                const wheelY = canvas.height * 0.5;

                if (results.landmarks && results.landmarks.length > 0) {{
                    for (let i = 0; i < results.landmarks.length; i++) {{
                        const marks = results.landmarks[i];
                        const isLeftHand = results.handedness[i][0].categoryName === "Right"; 
                        
                        ctx.fillStyle = isLeftHand ? "#00ffcc" : "#ff3366";
                        for(let j=0; j<marks.length; j++) {{
                            ctx.beginPath();
                            ctx.arc((1 - marks[j].x)*canvas.width, marks[j].y*canvas.height, 4, 0, 2*Math.PI);
                            ctx.fill();
                        }}

                        const wrist = marks[0];
                        const indexTip = marks[8];

                        let pxWrist = {{ x: (1 - wrist.x)*canvas.width, y: wrist.y*canvas.height }};
                        let pxIndexTip = {{ x: (1 - indexTip.x)*canvas.width, y: indexTip.y*canvas.height }};
                        const angle = getAngle(pxWrist, pxIndexTip);

                        if (isLeftHand) {{
                            currentRootIndex = Math.floor((angle + 15) / 30) % 12;
                            if (currentRootIndex < 0) currentRootIndex += 12;

                            if (onnxSession) {{
                                const flatLandmarks = new Float32Array(63);
                                for(let j=0; j<21; j++) {{
                                    flatLandmarks[j*3] = marks[j].x - wrist.x;
                                    flatLandmarks[j*3+1] = marks[j].y - wrist.y;
                                    flatLandmarks[j*3+2] = marks[j].z - wrist.z;
                                }}
                                const tensor = new ort.Tensor('float32', flatLandmarks, [1, 63]);
                                const onnxResults = await onnxSession.run({{ float_input: tensor }});
                                if (Number(onnxResults.output_label.data[0]) === 0) shouldPlay = true;
                            }}
                        }} else {{
                            currentChordIndex = Math.floor((angle + 22.5) / 45) % 8;
                            if (currentChordIndex < 0) currentChordIndex += 8;
                        }}
                    }}
                }}
                
                drawFixedRadialMenu(leftWheelX, wheelY, wheelRadius, 12, noteNames, currentRootIndex, shouldPlay ? noteNames[currentRootIndex] : "OFF", shouldPlay);
                drawFixedRadialMenu(rightWheelX, wheelY, wheelRadius, 8, chordTypes.map(c=>c.name), currentChordIndex, chordTypes[currentChordIndex].name, shouldPlay);
                
                if (currentRootIndex !== lastRootIndex || currentChordIndex !== lastChordIndex || shouldPlay !== lastShouldPlay) {{
                    playChord(currentRootIndex, currentChordIndex, shouldPlay);
                    lastRootIndex = currentRootIndex;
                    lastChordIndex = currentChordIndex;
                    lastShouldPlay = shouldPlay;
                }}
            }}
            window.requestAnimationFrame(predictWebcam);
        }}
        
        window.addEventListener('resize', () => {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }});
    </script>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #050505; font-family: 'Segoe UI', Tahoma, sans-serif; }}
        #webcam {{ display: block; position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; object-fit: cover; transform: scaleX(-1); z-index: 1; opacity: 0.6; }}
        #output_canvas {{ position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 2; pointer-events: none; }}
        
        #overlayBtn {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                       padding: 20px 50px; font-size: 20px; cursor: pointer; background: rgba(10, 10, 10, 0.9); 
                       color: #00ffcc; border: 2px solid #00ffcc; border-radius: 12px; z-index: 10; 
                       text-transform: uppercase; font-weight: bold; transition: 0.3s; box-shadow: 0 0 20px rgba(0,255,204,0.4); }}
        #overlayBtn:hover {{ background: #00ffcc; color: black; box-shadow: 0 0 40px #00ffcc; }}

        .bottom-bar {{
            position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
            background: rgba(20, 20, 20, 0.85); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
            display: flex; align-items: center; padding: 15px 30px; gap: 25px;
            color: #ccc; font-size: 14px; box-shadow: 0px 10px 30px rgba(0,0,0,0.8); z-index: 5; backdrop-filter: blur(10px);
        }}
        .bar-item {{ display: flex; align-items: center; gap: 10px; font-weight: 500;}}
        select, input[type="checkbox"] {{ background: #333; color: white; border: 1px solid #555; padding: 6px 10px; border-radius: 6px; outline: none; }}
        select:focus {{ border-color: #00ffcc; }}
    </style>
</head>
<body>
    <video id="webcam" autoplay playsinline></video>
    <canvas id="output_canvas"></canvas>
    <button id="overlayBtn">Initialize Synth</button>
    
    <div class="bottom-bar">
        <div class="bar-item"><span>Mode</span><select><option>Two-hand Chord</option></select></div>
        <div class="bar-item"><input type="checkbox" checked><span>Snap</span></div>
        <div class="bar-item"><span>Scale</span><select><option>Major</option></select></div>
        <div class="bar-item"><span>Wave</span><select><option>{st.session_state.wave_type}</option></select></div>
    </div>
</body>
</html>
"""

single_line_html = html_code.replace('\n', ' ').replace('\r', '')
safe_html = single_line_html.replace('"', '&quot;')

iframe_code = f'<iframe srcdoc="{safe_html}" width="100%" height="900px" allow="camera; microphone; autoplay;" style="border:none; overflow:hidden;"></iframe>'
st.markdown(iframe_code, unsafe_allow_html=True)
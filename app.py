import streamlit as st
import streamlit.components.v1 as components
import base64
import os
import numpy as np
import librosa
from streamlit_webrtc import webrtc_streamer,AudioProcessorBase,WebRtcMode
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide",page_title="AI Radial Synth")

st.markdown("""
    <style>
        #MainMenu {visibility:hidden;}
        header {visibility:hidden;}
        footer {visibility:hidden;}
        .block-container {padding:0;max-width:100%;}
    </style>
""",unsafe_allow_html=True)

if 'wave_type' not in st.session_state:
    st.session_state.wave_type="sine"

@st.cache_resource
def load_audio_model():
    model_path='timbre_bilstm.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None
bilstm_model=load_audio_model()

class TimbreAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer=np.array([])
        self.sample_rate=16000
        self.model=bilstm_model
    def recv(self,frame):
        audio_chunk=frame.to_ndarray()
        self.audio_buffer=np.append(self.audio_buffer,audio_chunk)
        if len(self.audio_buffer)>=self.sample_rate:
            audio_data=self.audio_buffer[:self.sample_rate].astype(np.float32)
            if self.model is not None:
                mfccs=librosa.feature.mfcc(y=audio_data,sr=self.sample_rate,n_mfcc=13)
                mfccs_processed=np.expand_dims(mfccs.T,axis=0)
                
                prediction=self.model.predict(mffcs_processed,verbose=0)
                class_id=np.argmax(prediction)

                if(class_id==0):
                    st.session_state.wave_type="sine"
                else:
                    st.session_state.wave_type="sawtooth"
            self.audio_buffer=np.array([])
        return frame
with st.container():
    st.markdown("<h4 style='color:white; text-align:center; position:absolute; z-index:100; top:10px; width:100%;'> Microphone: Timbre Classification Active</h4>",unsafe_allow_html=True)
    webrtc_streamer(
        key="timbre_classifier",
        mode= WebRtcMode.SENDONLY,
        audio_processor_factory=TimbreAudioProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

model_path_onnx="gesture_model.onnx"
onnx_base64=""
if os.path.exists(model_path_onnx):
    with open(model_path_onnx,"rb") as f:
        onnx_base64=base64.b64encode(f.read()).decode('utf-8')
else:
    st.error(f"Vision Model not found. Ensure '{model_path_onnx}' is present.")

html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import {{ HandLandmarker, FilesetResolver }} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs";
        import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.mjs";

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

        function drawFixedRadialMenu(cx, cy, radius, numSlices, labels, activeIndex, centerText) {{
            ctx.fillStyle = "rgba(40, 43, 48, 0.85)";
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
            ctx.fill();

            const sliceAngle = (2 * Math.PI) / numSlices;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.font = "14px Arial";

            for (let i = 0; i < numSlices; i++) {{
                const startAngle = i * sliceAngle - Math.PI / 2 - sliceAngle / 2;
                const endAngle = startAngle + sliceAngle;

                if (i === activeIndex) {{
                    ctx.fillStyle = numSlices === 8 ? "rgba(120, 40, 50, 0.6)" : "rgba(255, 255, 255, 0.1)";
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.arc(cx, cy, radius, startAngle, endAngle);
                    ctx.fill();
                }}

                ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx + Math.cos(startAngle) * radius, cy + Math.sin(startAngle) * radius);
                ctx.stroke();

                const textAngle = startAngle + sliceAngle / 2;
                const textX = cx + Math.cos(textAngle) * radius * 0.75;
                const textY = cy + Math.sin(textAngle) * radius * 0.75;
                ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
                ctx.fillText(labels[i], textX, textY);
            }}

            const innerRadius = radius * 0.35;
            ctx.fillStyle = "rgba(18, 18, 18, 1.0)";
            ctx.beginPath();
            ctx.arc(cx, cy, innerRadius, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
            ctx.font = "16px Arial";
            ctx.fillText(centerText, cx, cy);
            
            ctx.strokeStyle = "rgba(255,255,255,0.05)";
            ctx.beginPath();
            ctx.arc(cx, cy, innerRadius, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
            ctx.stroke();
        }}

        overlayBtn.addEventListener('click', async () => {{
            overlayBtn.innerText = "Loading AI Models...";
            initAudio();

            try {{
                const b64 = "{onnx_base64}";
                if (b64.length > 10) {{
                    const modelBytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                    onnxSession = await ort.InferenceSession.create(modelBytes);
                }}

                const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
                handLandmarker = await HandLandmarker.createFromOptions(vision, {{
                    baseOptions: {{
                        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                        delegate: "GPU"
                    }},
                    runningMode: "VIDEO",
                    numHands: 2
                }});

                const stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
                video.srcObject = stream;
                video.addEventListener('loadeddata', predictWebcam);
                overlayBtn.style.display = "none";
            }} catch (error) {{
                overlayBtn.innerText = "Error loading models.";
                console.error(error);
            }}
        }});

        let lastVideoTime = -1;
        let currentRootIndex = 0; 
        let currentChordIndex = 0;

        async function predictWebcam() {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            let startTimeMs = performance.now();
            
            if (lastVideoTime !== video.currentTime) {{
                lastVideoTime = video.currentTime;
                const results = handLandmarker.detectForVideo(video, startTimeMs);
                
                ctx.save();
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);

                let shouldPlay = false;

                const wheelRadius = Math.min(canvas.width, canvas.height) * 0.35;
                const leftWheelX = canvas.width * 0.25;
                const rightWheelX = canvas.width * 0.75;
                const wheelY = canvas.height * 0.45;

                if (results.landmarks && results.landmarks.length > 0) {{
                    for (let i = 0; i < results.landmarks.length; i++) {{
                        const marks = results.landmarks[i];
                        const isLeftHand = results.handedness[i][0].categoryName === "Right";
                        const wrist = marks[0];
                        const indexTip = marks[8];

                        const angle = getAngle(wrist, indexTip);

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
                
                drawFixedRadialMenu(leftWheelX, wheelY, wheelRadius, 12, noteNames, currentRootIndex, shouldPlay ? noteNames[currentRootIndex] : "OFF");
                drawFixedRadialMenu(rightWheelX, wheelY, wheelRadius, 8, chordTypes.map(c=>c.name), currentChordIndex, chordTypes[currentChordIndex].name);
                
                playChord(currentRootIndex, currentChordIndex, shouldPlay);
                ctx.restore();
            }}
            window.requestAnimationFrame(predictWebcam);
        }}
        
        window.addEventListener('resize', () => {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }});
    </script>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #000; font-family: sans-serif; }}
        video {{ display: none; }}
        canvas {{ display: block; width: 100vw; height: 100vh; object-fit: cover; }}
        
        #overlayBtn {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                       padding: 20px 40px; font-size: 24px; cursor: pointer; background: rgba(0,0,0,0.8); 
                       color: white; border: 2px solid white; border-radius: 10px; z-index: 10; }}
        #overlayBtn:hover {{ background: rgba(50,50,50,0.8); }}

        .bottom-bar {{
            position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
            background-color: rgba(30, 32, 36, 0.9); border-radius: 10px;
            display: flex; align-items: center; padding: 10px 25px; gap: 20px;
            color: #ccc; font-size: 14px; font-family: Arial;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.5); z-index: 5;
        }}
        .bar-item {{ display: flex; align-items: center; gap: 8px; }}
        select, input[type="checkbox"] {{ background: #333; color: white; border: none; padding: 5px; border-radius: 4px; }}
    </style>
</head>
<body>
    <button id="overlayBtn">Start Application</button>
    <video id="webcam" autoplay playsinline></video>
    <canvas id="output_canvas"></canvas>
    
    <div class="bottom-bar">
        <div class="bar-item">
            <span>Mode</span>
            <select><option>Two-hand Chord</option></select>
        </div>
        <div class="bar-item">
            <input type="checkbox" checked> <span>Snap</span>
        </div>
        <div class="bar-item">
            <input type="checkbox"> <span>Simple (ABCDEFG)</span>
        </div>
        <div class="bar-item">
            <span>Scale</span>
            <select><option>Major</option></select>
        </div>
        <div class="bar-item">
            <span>Wave</span>
            <!-- Dynamically updating based on Bi-LSTM classification -->
            <select><option>{st.session_state.wave_type}</option></select>
        </div>
        <div class="bar-item">
            <span>Range</span>
            <select><option>3 oct</option></select>
        </div>
    </div>
</body>
</html>
"""

components.html(html_code, height=900)
let mediaRecorder;
let audioChunks = [];

let audioContext;
let analyser;
let dataArray;
let stream;

// audio output
let queue = [];
let playing = false;
let audioCtx = new AudioContext({ sampleRate: 24000 });

const micIcon = document.getElementById("mic");
const output = document.getElementById("output");
const ws = new WebSocket(`ws://${window.location.host}/ws`);
const wsAudio = new WebSocket(`ws://${window.location.host}/ws_audio`);
const wsTTS = new WebSocket(`ws://${window.location.host}/ws_tts`);

ws.onopen = () => {
    console.log("WebSocket connected");
};

ws.onmessage = (event) => {
    const bc_result = parseFloat(event.data) * 100;

    document.getElementById("barChart").innerHTML =
        `<p>p_bc: ${bc_result.toFixed(1)}%</p>`;
    myChart.data.datasets[0].data = [
        bc_result
    ];

    myChart.data.datasets[0].backgroundColor = bc_result > 50
        ? 'rgb(255, 190, 105)'
        : 'rgba(169, 255, 151, 0.7)'

    myChart.update();
};

ws.onclose = () => {
    console.log("WebSocket disconnected");
};

wsAudio.onopen = async () => {
    console.log("wsAudio connected");
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext({ sampleRate: 16000 });
    await audioContext.audioWorklet.addModule("/frontend/worklet-processor.js");
    const source = audioContext.createMediaStreamSource(stream);
    const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");
    source.connect(workletNode);
    workletNode.connect(audioContext.destination);

    workletNode.port.onmessage = (event) => {
        if (wsAudio.readyState !== 1) return;
        const float32 = event.data;
        wsAudio.send(float32.buffer);
    };
};


wsTTS.onmessage = async (event) => {
    const buffer = await event.data.arrayBuffer();
    const msg = new Float32Array(buffer);

    queue.push(msg);

    if (!playing) playNext();
};

async function playNext() {
    if (queue.length === 0) {
        playing = false;
        return;
    }

    playing = true;

    const msg = queue.shift();

    const audioBuffer = audioCtx.createBuffer(
        1,
        msg.length,
        audioCtx.sampleRate
    );

    audioBuffer.copyToChannel(msg, 0);

    const source = audioCtx.createBufferSource();
    const gain = audioCtx.createGain();

    source.buffer = audioBuffer;

    source.connect(gain);
    gain.connect(audioCtx.destination);

    const start = audioCtx.currentTime;

    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.linearRampToValueAtTime(1.0, start + 0.003);
    gain.gain.linearRampToValueAtTime(1.0, start + audioBuffer.duration - 0.003);
    gain.gain.linearRampToValueAtTime(0.0001, start + audioBuffer.duration);

    source.start();

    source.onended = () => {
        playNext();
    };
}

async function initMic() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = async (e) => {
        if (e.data.size > 0) {
            audioChunks.push(e.data);
            const formData = new FormData();
            formData.append("file", e.data, "chunk.wav");
            try {
                const resEmotion = await fetch("/emotion_recognition", {
                    method: "POST",
                    body: formData
                });
            } catch (err) {
                console.error("Emotion recognition failed:", err);
            }
        }
    };

    mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: "audio/wav" });
        audioChunks = [];
        const formData = new FormData();
        formData.append("file", blob, "speech.wav");
        try {
            const res = await fetch("/audio", { method: "POST", body: formData });
            const data = await res.json();
            output.innerHTML = `<p><b>You:</b> ${data.transcript}</p>
                                <p><b>Assistant:</b> ${data.response}</p>`;
        } catch (err) {
            console.error(err);
            output.innerHTML = "<p style='color:red'>Audio upload failed</p>";
        }
    };

    audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    dataArray = new Uint8Array(analyser.fftSize);
}

initMic();

document.addEventListener("keydown", e => {
    if (e.code === "Space" && mediaRecorder.state !== "recording") {
        audioChunks = [];
        mediaRecorder.start();
        micIcon.classList.add("recording");
    }
    sendMicState({
        type: "mic",
        state: "start",
        ts: Date.now()
    });
});

document.addEventListener("keyup", e => {
    if (e.code === "Space" && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        micIcon.classList.remove("recording");
    }
    sendMicState({
        type: "mic",
        state: "stop",
        ts: Date.now()
    });
});


function sendMicState(msg) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
    }
}


// BarChart
const ctx = document.getElementById('barChart').getContext('2d');

const data = {
    labels: ['Backchannel Prediction [%]'],
    datasets: [{
        label: 'Score',
        data: [0],
        backgroundColor: 'rgba(0, 255, 136, 0.7)',
        borderColor: 'rgba(0, 255, 136, 1)',
        borderWidth: 1,
        borderRadius: 6,
        maxBarThickness: 60
    }]
};

const config = {
    type: 'bar',
    data: data,
    options: {
        indexAxis: 'y',

        responsive: true,
        plugins: {
            legend: { display: false },

            annotation: {
                annotations: {
                    thresholdLine: {
                        type: 'line',
                        xMin: 60,
                        xMax: 60,
                        borderColor: 'rgb(255, 233, 33)',
                        borderWidth: 2,
                        label: {
                            display: true,
                            content: 'Threshold',
                            position: 'start',
                            color: 'rgb(255, 246, 162)'
                        }
                    }
                }
            }
        },

        scales: {
            x: {
                min: 0,
                max: 100,
                ticks: {
                    color: '#fff',
                    stepSize: 20
                }
            },
            y: {
                ticks: { color: '#fff' }
            }
        }
    }
};

const myChart = new Chart(ctx, config);

let mediaRecorder;
let audioChunks = [];

let audioContext;
let analyser;
let dataArray;

// const canvas = document.getElementById("wave");
// const ctx = canvas.getContext("2d");

const micIcon = document.getElementById("mic");
const output = document.getElementById("output");
const ws = new WebSocket(`ws://${window.location.host}/ws`);

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

async function initMic() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    // mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
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
    // drawWave();
}


initMic();

document.addEventListener("keydown", e => {
    if (e.code === "Space" && mediaRecorder.state !== "recording") {
        audioChunks = [];
        mediaRecorder.start();
        micIcon.classList.add("recording");
    }
});

document.addEventListener("keyup", e => {
    if (e.code === "Space" && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        micIcon.classList.remove("recording");
    }
});
function updateBars(values) {

    values.forEach((v, i) => {

        const bar = document.getElementById("bar" + (i + 1))
        bar.style.height = (v * 200) + "px"

    })
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

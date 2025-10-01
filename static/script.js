document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');
    const startButton = document.getElementById('start-button');
    const stopButton = document.getElementById('stop-button');
    const logOutput = document.getElementById('log-output');
    const progressBar = document.getElementById('progress-bar');
    const progressLabel = document.getElementById('progress-label');
    
    let uploadedFiles = [];

    // --- Log Helper ---
    function log(level, message) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-${level.toLowerCase()}`;
        logEntry.textContent = `[${timestamp}] [${level}] ${message}`;
        logOutput.appendChild(logEntry);
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    // --- Form Handling ---
    uploadForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const formData = new FormData(uploadForm);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                uploadStatus.textContent = `Fehler: ${data.error}`;
                log('ERROR', `Upload fehlgeschlagen: ${data.error}`);
            } else {
                uploadStatus.textContent = data.message;
                log('INFO', data.message);
                uploadedFiles = data.files;
                startButton.disabled = false;
            }
        })
        .catch(error => {
            uploadStatus.textContent = 'Upload fehlgeschlagen.';
            log('ERROR', `Upload-Anfrage fehlgeschlagen: ${error}`);
            console.error('Upload error:', error);
        });
    });

    // --- Socket.IO Event Handlers ---
    socket.on('connect', () => {
        console.log('Verbunden mit dem Server.');
        log('INFO', 'Mit dem Server verbunden.');
    });

    socket.on('log', (data) => {
        log(data.level, data.message);
    });

    socket.on('progress', (data) => {
        const percent = (data.current / data.total) * 100;
        progressBar.style.width = `${percent}%`;
        progressLabel.textContent = `Phase: ${data.stage} (${data.current} von ${data.total})`;
    });
    
    socket.on('pipeline_finished', () => {
        startButton.disabled = false;
        stopButton.disabled = true;
        progressLabel.textContent = "Verarbeitung abgeschlossen oder gestoppt.";
    });

    // --- Button Clicks ---
    startButton.addEventListener('click', () => {
        if (uploadedFiles.length === 0) {
            log('WARN', 'Keine Dateien hochgeladen, um die Verarbeitung zu starten.');
            return;
        }
        
        const settingsForm = document.getElementById('settings-form');
        const settings = {
            files: uploadedFiles,
            min_sec: document.getElementById('min_sec').value,
            max_sec: document.getElementById('max_sec').value,
            min_dbfs: document.getElementById('min_dbfs').value,
        };

        socket.emit('start_processing', settings);
        startButton.disabled = true;
        stopButton.disabled = false;
        progressBar.style.width = '0%';
        progressLabel.textContent = 'Starte...';
    });

    stopButton.addEventListener('click', () => {
        socket.emit('stop_processing');
        stopButton.disabled = true; // Disable until process confirms stop
    });
});

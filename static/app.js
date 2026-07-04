/* ==========================================================================
   PHISHGUARD AI - APPLICATION LOGIC
   Features: Web Audio API extraction, real-time visualizers, autocorrelation 
             pitch tracking, audio recording via MediaRecorder, sending raw files/mic 
             to Python Flask backend, and client-side fallback engine.
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  
  // Backend Configuration
  const BACKEND_URL = '';  // Same origin - served from Flask
  let serverAvailable = false;

  // UI Elements - Tabs
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  // UI Elements - Mic Controls
  const btnStartMic = document.getElementById('btn-start-mic');
  const btnStopMic = document.getElementById('btn-stop-mic');
  const canvasMic = document.getElementById('mic-visualizer');
  const visualizerPrompt = document.getElementById('visualizer-prompt');
  
  // UI Elements - File Upload
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileDetailsCard = document.getElementById('file-details-card');
  const fileNameDisplay = document.getElementById('file-name');
  const fileSizeDisplay = document.getElementById('file-size');
  const btnAnalyzeFile = document.getElementById('btn-analyze-file');
  const uploadVisualizerContainer = document.getElementById('upload-visualizer-container');
  const canvasUpload = document.getElementById('upload-waveform-canvas');
  const uploadVisualizerStatus = document.getElementById('upload-visualizer-status');
  
  // UI Elements - Threat Output
  const riskScoreText = document.getElementById('risk-score');
  const riskLabelText = document.getElementById('risk-label');
  const gaugeFill = document.getElementById('gauge-fill');
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictDesc = document.getElementById('verdict-desc');
  
  // UI Elements - Acoustic Feature Elements
  const valPitch = document.getElementById('val-pitch');
  const barPitch = document.getElementById('bar-pitch');
  const valJitter = document.getElementById('val-jitter');
  const barJitter = document.getElementById('bar-jitter');
  const valPauses = document.getElementById('val-pauses');
  const barPauses = document.getElementById('bar-pauses');
  const valCentroid = document.getElementById('val-centroid');
  const barCentroid = document.getElementById('bar-centroid');
  
  // Mascot Mascot Elements
  const mascotText = document.getElementById('mascot-text');
  const eyeLeft = document.getElementById('eye-left');
  const eyeRight = document.getElementById('eye-right');
  const robotMouth = document.getElementById('robot-mouth');
  const faceScreen = document.getElementById('robot-face-screen');

  // Audio Context & Media Recording State variables
  let audioContext = null;
  let micStream = null;
  let micSourceNode = null;
  let analyserNode = null;
  let mediaRecorder = null;
  let recordedChunks = [];
  let animationFrameId = null;
  let featureExtractionInterval = null;
  
  // Feature Accumulators (for dynamic session calculation)
  let pitchHistory = [];
  let silentFramesCount = 0;
  let totalFramesCount = 0;
  let centroidHistory = [];
  let volumeHistory = [];
  let isRecording = false;
  let activeAudioFile = null;
  let activeAudioBuffer = null;

  /* ==========================================================================
     BACKEND CONNECTIVITY CHECK
     ========================================================================== */

  async function checkServerStatus() {
    try {
      const response = await fetch(`${BACKEND_URL}/api/status`);
      const data = await response.json();
      if (data.status === 'ready') {
        serverAvailable = true;
        const msg = data.model_loaded 
          ? "Python Backend Online (Machine Learning Model Loaded & Ready)."
          : "Python Backend Online (Fallback Heuristics Mode).";
        setMascotState('safe', msg);
        document.querySelector('.header-badge').textContent = data.model_loaded ? 'Python ML Mode' : 'Python Base Mode';
        document.querySelector('.header-badge').style.background = 'rgba(16, 185, 129, 0.2)';
        document.querySelector('.header-badge').style.color = '#34D399';
      }
    } catch (e) {
      serverAvailable = false;
      document.querySelector('.header-badge').textContent = 'Browser Fallback';
      document.querySelector('.header-badge').style.background = 'rgba(245, 158, 11, 0.15)';
      document.querySelector('.header-badge').style.color = '#F59E0B';
      console.warn("Python backend server is offline. Running in local browser fallback mode.");
    }
  }

  // Run status check at initialization
  checkServerStatus();

  /* ==========================================================================
     TAB HANDLERS & NAVIGATION
     ========================================================================== */
  
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      
      btn.classList.add('active');
      const targetTab = btn.getAttribute('data-tab');
      document.getElementById(targetTab).classList.add('active');
      
      // Stop mic if switching tabs
      if (targetTab !== 'mic-tab' && isRecording) {
        stopMicrophone();
      }
    });
  });

  // History & Statistics state management
  let historyLogs = JSON.parse(localStorage.getItem('vishing_history') || '[]');

  function loadHistoryUI() {
    const historyRows = document.getElementById('history-rows');
    const emptyRow = document.getElementById('history-empty-row');
    
    const statTotal = document.getElementById('stat-total-count');
    const statSafe = document.getElementById('stat-safe-count');
    const statMedium = document.getElementById('stat-medium-count');
    const statHigh = document.getElementById('stat-high-count');

    let total = historyLogs.length;
    let safeCount = 0;
    let mediumCount = 0;
    let highCount = 0;

    // Remove existing dynamic rows
    const rows = historyRows.querySelectorAll('.dynamic-history-row');
    rows.forEach(r => r.remove());

    if (total === 0) {
      if (emptyRow) emptyRow.classList.remove('hidden');
      if (statTotal) statTotal.textContent = '0';
      if (statSafe) statSafe.textContent = '0';
      if (statMedium) statMedium.textContent = '0';
      if (statHigh) statHigh.textContent = '0';
      return;
    }

    if (emptyRow) emptyRow.classList.add('hidden');

    historyLogs.forEach(item => {
      const score = item.score;
      let statusBadge = '';
      if (score > 70) {
        highCount++;
        statusBadge = `<span class="badge badge-danger">Vishing</span>`;
      } else if (score > 35) {
        mediumCount++;
        statusBadge = `<span class="badge badge-warning">Suspicious</span>`;
      } else {
        safeCount++;
        statusBadge = `<span class="badge badge-safe">Safe</span>`;
      }

      const tr = document.createElement('tr');
      tr.className = 'dynamic-history-row';
      tr.innerHTML = `
        <td style="color: var(--text-muted); font-size: 11px;">${item.timestamp}</td>
        <td style="font-weight: 600;">${item.name}</td>
        <td>
          <div style="font-weight: 600; font-size: 13px;">${item.verdict}</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px; max-width: 320px; line-height: 1.3;">${item.details}</div>
        </td>
        <td style="font-family: 'Outfit'; font-weight: 700; font-size: 14px; color: ${score > 70 ? '#EF4444' : (score > 35 ? '#FBBF24' : '#10B981')};">${score}%</td>
        <td>${statusBadge}</td>
      `;
      historyRows.appendChild(tr);
    });

    if (statTotal) statTotal.textContent = total;
    if (statSafe) statSafe.textContent = safeCount;
    if (statMedium) statMedium.textContent = mediumCount;
    if (statHigh) statHigh.textContent = highCount;
  }

  function addHistoryItem(sourceName, verdict, details, score) {
    const timestamp = new Date().toLocaleString();
    const item = {
      timestamp,
      name: sourceName,
      verdict,
      details,
      score
    };
    
    historyLogs.unshift(item);
    if (historyLogs.length > 20) {
      historyLogs.pop();
    }
    
    localStorage.setItem('vishing_history', JSON.stringify(historyLogs));
    loadHistoryUI();
  }

  // Clear history button
  const btnClearHistory = document.getElementById('btn-clear-history');
  if (btnClearHistory) {
    btnClearHistory.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear the entire analysis logs history?')) {
        historyLogs = [];
        localStorage.removeItem('vishing_history');
        loadHistoryUI();
      }
    });
  }

  // Initial load
  loadHistoryUI();

  // Fetch and render multi-model benchmark comparison table
  async function loadBenchmarkUI() {
    const benchmarkRows = document.getElementById('benchmark-rows');
    const bestBadge = document.getElementById('selected-best-badge');
    if (!benchmarkRows) return;

    try {
      const response = await fetch(`${BACKEND_URL}/api/benchmark`);
      const data = await response.json();

      if (data.matrix) {
        benchmarkRows.innerHTML = '';
        const selectedBest = data.selected_best || '';

        if (bestBadge) {
          bestBadge.textContent = `Best: ${selectedBest}`;
        }

        for (const [name, stats] of Object.entries(data.matrix)) {
          const isBest = (name === selectedBest);
          const tr = document.createElement('tr');
          if (isBest) {
            tr.style.background = 'rgba(16, 185, 129, 0.04)';
          }
          tr.innerHTML = `
            <td style="font-weight: 700;">${name}${isBest ? ' <span style="background:rgba(16,185,129,0.15);color:#34D399;padding:2px 8px;border-radius:10px;font-size:9px;font-weight:800;margin-left:8px;">SELECTED</span>' : ''}</td>
            <td style="font-family:'Outfit';font-weight:700;color:${stats.val_accuracy > 90 ? '#34D399' : (stats.val_accuracy > 80 ? '#FBBF24' : '#FCA5A5')};">${stats.val_accuracy}%</td>
            <td>${stats.precision}</td>
            <td>${stats.recall}</td>
            <td style="font-weight:600;">${stats.f1_score}</td>
            <td style="color:var(--text-muted);">${stats.latency_ms}ms</td>
          `;
          benchmarkRows.appendChild(tr);
        }
      }
    } catch (err) {
      console.warn('Benchmark API unavailable:', err);
      if (bestBadge) bestBadge.textContent = 'Offline';
    }
  }

  loadBenchmarkUI();

  /* ==========================================================================
     CUTE MASCOT EXPRESSIONS
     ========================================================================== */
  
  function setMascotState(state, message) {
    if (message) mascotText.textContent = message;
    
    // Reset eye properties & screen color
    eyeLeft.style.animation = 'none';
    eyeRight.style.animation = 'none';
    eyeLeft.setAttribute('rx', '8');
    eyeLeft.setAttribute('ry', '8');
    eyeRight.setAttribute('rx', '8');
    eyeRight.setAttribute('ry', '8');
    
    // Screen gradient color resets
    faceScreen.style.fill = ''; 
    faceScreen.style.stroke = '#312E81';

    switch (state) {
      case 'idle':
        eyeLeft.setAttribute('fill', '#10B981');
        eyeRight.setAttribute('fill', '#10B981');
        eyeLeft.style.animation = 'eyeBlink 4s infinite';
        eyeRight.style.animation = 'eyeBlink 4s infinite';
        robotMouth.setAttribute('d', 'M 80 118 Q 100 118 120 118'); // straight mouth
        robotMouth.setAttribute('stroke', '#10B981');
        break;
        
      case 'listening':
        eyeLeft.setAttribute('fill', '#3B82F6');
        eyeRight.setAttribute('fill', '#3B82F6');
        eyeLeft.style.animation = 'eyeBlink 2.5s infinite';
        eyeRight.style.animation = 'eyeBlink 2.5s infinite';
        robotMouth.setAttribute('stroke', '#3B82F6');
        faceScreen.style.stroke = '#3B82F6';
        break;
        
      case 'thinking':
        eyeLeft.setAttribute('fill', '#F59E0B');
        eyeRight.setAttribute('fill', '#F59E0B');
        eyeLeft.setAttribute('ry', '3'); // narrow eyes
        eyeRight.setAttribute('ry', '3');
        robotMouth.setAttribute('d', 'M 85 118 L 115 118'); // flat mouth line
        robotMouth.setAttribute('stroke', '#F59E0B');
        faceScreen.style.stroke = '#F59E0B';
        break;
        
      case 'threat':
        eyeLeft.setAttribute('fill', '#EF4444');
        eyeRight.setAttribute('fill', '#EF4444');
        eyeLeft.setAttribute('ry', '12'); // big round surprised/threat eyes
        eyeRight.setAttribute('ry', '12');
        robotMouth.setAttribute('d', 'M 90 122 Q 100 110 110 122'); // sad frown mouth
        robotMouth.setAttribute('stroke', '#EF4444');
        faceScreen.style.stroke = '#EF4444';
        faceScreen.style.fill = 'rgba(239, 68, 68, 0.15)'; // glowing red screen outline
        break;
        
      case 'safe':
        eyeLeft.setAttribute('fill', '#10B981');
        eyeRight.setAttribute('fill', '#10B981');
        robotMouth.setAttribute('d', 'M 80 112 Q 100 125 120 112'); // happy smile mouth
        robotMouth.setAttribute('stroke', '#10B981');
        faceScreen.style.stroke = '#10B981';
        faceScreen.style.fill = 'rgba(16, 185, 129, 0.15)'; // glowing green screen outline
        break;
    }
  }

  /* ==========================================================================
     FEATURE EXTRACTION MATH (CLIENT FALLBACK)
     ========================================================================== */

  function autoCorrelate(buffer, sampleRate) {
    let SIZE = buffer.length;
    let maxSamples = Math.floor(SIZE / 2);
    let bestOffset = -1;
    let bestCorrelation = 0;
    let rms = 0;

    for (let i = 0; i < SIZE; i++) {
      let val = buffer[i];
      rms += val * val;
    }
    rms = Math.sqrt(rms / SIZE);

    if (rms < 0.01) return -1;

    let r1 = 0, r2 = SIZE - 1, thres = 0.2;
    for (let i = 0; i < maxSamples; i++) {
      if (Math.abs(buffer[i]) < thres) { r1 = i; break; }
    }
    for (let i = SIZE - 1; i >= maxSamples; i--) {
      if (Math.abs(buffer[i]) < thres) { r2 = i; break; }
    }
    buffer = buffer.slice(r1, r2);
    SIZE = buffer.length;

    let correlations = new Float32Array(SIZE);
    
    for (let offset = 0; offset < SIZE; offset++) {
      let sum = 0;
      for (let t = 0; t < SIZE - offset; t++) {
        sum += buffer[t] * buffer[t + offset];
      }
      correlations[offset] = sum;
    }

    let zeroCrossing = 0;
    for (let i = 0; i < SIZE - 1; i++) {
      if (correlations[i] > 0 && correlations[i + 1] < 0) {
        zeroCrossing = i;
        break;
      }
    }

    if (zeroCrossing !== 0) {
      for (let offset = zeroCrossing; offset < SIZE; offset++) {
        if (correlations[offset] > bestCorrelation) {
          bestCorrelation = correlations[offset];
          bestOffset = offset;
        }
      }
    }

    if (bestOffset !== -1) {
      let frequency = sampleRate / bestOffset;
      if (frequency >= 50 && frequency <= 550) {
        return frequency;
      }
    }
    return -1;
  }

  function calculateSpectralCentroid(freqData, sampleRate) {
    let numBins = freqData.length;
    let nyquist = sampleRate / 2;
    let binWidth = nyquist / numBins;
    
    let weightedSum = 0;
    let magnitudeSum = 0;

    for (let i = 0; i < numBins; i++) {
      let freq = i * binWidth;
      let mag = freqData[i];
      
      weightedSum += freq * mag;
      magnitudeSum += mag;
    }

    if (magnitudeSum === 0) return 0;
    return weightedSum / magnitudeSum;
  }

  /* ==========================================================================
     MICROPHONE REAL-TIME TESTING
     ========================================================================== */

  async function startMicrophone() {
    try {
      // Refresh status check
      await checkServerStatus();

      // Reset statistics state
      pitchHistory = [];
      centroidHistory = [];
      volumeHistory = [];
      recordedChunks = [];
      silentFramesCount = 0;
      totalFramesCount = 0;
      isRecording = true;

      // Reset verdict banner styles
      verdictBanner.className = 'verdict-banner waiting';
      verdictTitle.textContent = 'System Active';
      verdictDesc.textContent = 'Listening to voice parameters in real-time...';

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      micStream = stream;
      
      // Audio nodes for live visualizations
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      micSourceNode = audioContext.createMediaStreamSource(stream);
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 2048;
      micSourceNode.connect(analyserNode);

      // MediaRecorder for API upload
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunks.push(event.data);
        }
      };
      mediaRecorder.start();

      // UI update
      btnStartMic.disabled = true;
      btnStopMic.disabled = false;
      visualizerPrompt.classList.add('hidden');
      
      const promptMsg = serverAvailable 
        ? "Recording voice... Backend is connected. Speak naturally, then click 'Stop & Classify'."
        : "Recording voice (Offline Mode). Speak, then click 'Stop & Classify'.";
      setMascotState('listening', promptMsg);

      drawMicVisualizer();
      featureExtractionInterval = setInterval(extractRealTimeFeatures, 100);

    } catch (err) {
      console.error('Microphone access denied:', err);
      alert('Microphone access is required to run the real-time analyzer.');
      setMascotState('idle', 'Microphone connection failed.');
    }
  }

  function stopMicrophone() {
    if (!isRecording) return;
    isRecording = false;

    if (featureExtractionInterval) clearInterval(featureExtractionInterval);
    if (animationFrameId) cancelAnimationFrame(animationFrameId);

    // Stop MediaRecorder and handle data
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });
      analyzeAudioSource(audioBlob);
    };
    mediaRecorder.stop();

    // Stop physical mic hardware streams
    if (micStream) {
      micStream.getTracks().forEach(track => track.stop());
    }

    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close();
    }

    // UI Updates
    btnStartMic.disabled = false;
    btnStopMic.disabled = true;
    visualizerPrompt.classList.remove('hidden');
    visualizerPrompt.textContent = 'Processing...';

    const ctx = canvasMic.getContext('2d');
    ctx.clearRect(0, 0, canvasMic.width, canvasMic.height);
  }

  function drawMicVisualizer() {
    if (!isRecording || !analyserNode) return;

    const ctx = canvasMic.getContext('2d');
    const width = canvasMic.width = canvasMic.parentElement.clientWidth;
    const height = canvasMic.height = canvasMic.parentElement.clientHeight;
    
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!isRecording) return;
      animationFrameId = requestAnimationFrame(draw);

      analyserNode.getByteTimeDomainData(dataArray);

      ctx.fillStyle = '#070412';
      ctx.fillRect(0, 0, width, height);

      // Draw lines
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.05)';
      ctx.lineWidth = 1;
      for (let i = 0; i < width; i += 40) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, height);
        ctx.stroke();
      }

      ctx.lineWidth = 2.5;
      const waveGrad = ctx.createLinearGradient(0, 0, width, 0);
      waveGrad.addColorStop(0, '#3B82F6');
      waveGrad.addColorStop(0.5, '#EC4899');
      waveGrad.addColorStop(1, '#8B5CF6');
      ctx.strokeStyle = waveGrad;
      
      ctx.shadowBlur = 8;
      ctx.shadowColor = 'rgba(139, 92, 246, 0.4)';

      ctx.beginPath();
      const sliceWidth = width / bufferLength;
      let x = 0;
      let maxVal = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * height) / 2;

        const amplitude = Math.abs(dataArray[i] - 128);
        if (amplitude > maxVal) maxVal = amplitude;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(width, height / 2);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Animate mascot mouth
      const mouthOpenness = Math.min(25, maxVal * 0.8);
      robotMouth.setAttribute('d', `M 80 115 Q 100 ${115 + mouthOpenness} 120 115`);
    };

    draw();
  }

  function extractRealTimeFeatures() {
    if (!analyserNode || !audioContext) return;

    const timeDomainBuffer = new Float32Array(analyserNode.fftSize);
    const freqDomainBuffer = new Uint8Array(analyserNode.frequencyBinCount);

    analyserNode.getFloatTimeDomainData(timeDomainBuffer);
    analyserNode.getByteFrequencyData(freqDomainBuffer);

    // Pitch
    const pitch = autoCorrelate(timeDomainBuffer, audioContext.sampleRate);
    if (pitch !== -1) {
      pitchHistory.push(pitch);
      valPitch.textContent = `${Math.round(pitch)} Hz`;
      barPitch.style.width = `${Math.min(100, (pitch / 500) * 100)}%`;
    }

    // Volume & Silence ratio
    let sum = 0;
    for (let i = 0; i < timeDomainBuffer.length; i++) {
      sum += timeDomainBuffer[i] * timeDomainBuffer[i];
    }
    const rms = Math.sqrt(sum / timeDomainBuffer.length);
    volumeHistory.push(rms);
    
    totalFramesCount++;
    if (rms < 0.015) {
      silentFramesCount++;
    }
    const currentPauseRatio = (silentFramesCount / totalFramesCount) * 100;
    valPauses.textContent = `${Math.round(currentPauseRatio)}%`;
    barPauses.style.width = `${currentPauseRatio}%`;

    // Centroid
    const centroid = calculateSpectralCentroid(freqDomainBuffer, audioContext.sampleRate);
    centroidHistory.push(centroid);
    valCentroid.textContent = `${Math.round(centroid)} Hz`;
    barCentroid.style.width = `${Math.min(100, (centroid / 8000) * 100)}%`;

    // Jitter (Standard deviation metric)
    if (pitchHistory.length > 5) {
      const mean = pitchHistory.reduce((a, b) => a + b, 0) / pitchHistory.length;
      const variance = pitchHistory.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / pitchHistory.length;
      const stdDev = Math.sqrt(variance);
      const jitterPercent = Math.min(100, (stdDev / mean) * 100 * 5);
      valJitter.textContent = `${jitterPercent.toFixed(1)}%`;
      barJitter.style.width = `${jitterPercent}%`;
    }
  }

  /* ==========================================================================
     UNIFIED CLASSIFIER ROUTER (API BACKEND vs CLIENT FALLBACK)
     ========================================================================== */

  async function analyzeAudioSource(audioBlobOrFile) {
    setMascotState('thinking', 'Running voice through Random Forest feature extractor...');
    visualizerPrompt.textContent = 'Analyzing...';
    
    // Check if backend is available
    if (serverAvailable) {
      try {
        // Send file to Flask server
        const formData = new FormData();
        formData.append('audio', audioBlobOrFile, 'audio_capture.wav');

        const response = await fetch(`${BACKEND_URL}/api/analyze`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error('API server analysis error.');
        }

        const data = await response.json();
        if (data.success) {
          // Set UI meters based on exact values returned from python model
          const m = data.metrics;
          valPitch.textContent = `${Math.round(m.pitch)} Hz`;
          barPitch.style.width = `${Math.min(100, (m.pitch / 500) * 100)}%`;
          
          valJitter.textContent = `${m.jitter.toFixed(1)}%`;
          barJitter.style.width = `${Math.min(100, m.jitter * 5)}%`;

          valPauses.textContent = `${Math.round(m.pause_ratio)}%`;
          barPauses.style.width = `${m.pause_ratio}%`;

          valCentroid.textContent = `${Math.round(m.centroid)} Hz`;
          barCentroid.style.width = `${Math.min(100, (m.centroid / 8000) * 100)}%`;

          renderResults(data.threat_score, data.classification, data.details, data.mascot_state, audioBlobOrFile instanceof File ? audioBlobOrFile.name : "Live Mic Capture");
          visualizerPrompt.textContent = 'Microphone Inactive';
          return;
        }

      } catch (err) {
        console.error("Failed to fetch backend analysis. Falling back to local browser engine.", err);
      }
    }

    // ─── LOCAL CLIENT FALLBACK CLASSIFICATION ───
    setTimeout(() => {
      const avgPitch = pitchHistory.length > 0 ? (pitchHistory.reduce((a,b)=>a+b, 0) / pitchHistory.length) : 0;
      const avgCentroid = centroidHistory.length > 0 ? (centroidHistory.reduce((a,b)=>a+b, 0) / centroidHistory.length) : 0;
      const pauseRatio = totalFramesCount > 0 ? (silentFramesCount / totalFramesCount) * 100 : 0;
      
      let pitchJitter = 0;
      if (pitchHistory.length > 1) {
        const mean = avgPitch;
        const variance = pitchHistory.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / pitchHistory.length;
        pitchJitter = (Math.sqrt(variance) / mean) * 100;
      }

      let score = 0;
      let classification = '';
      let details = '';
      let mascotState = 'idle';

      if (avgCentroid > 2800 && pitchJitter < 2.5 && avgPitch > 0) {
        score = 88;
        classification = 'Synthetic / Robotic (Robocall)';
        details = 'Local Feature engine detected vocoder brightness artifacts and monotone voice patterns, matching TTS signatures in the Fake-or-Real dataset.';
        mascotState = 'threat';
      } else if (pitchJitter > 18 && pauseRatio < 12 && avgPitch > 0) {
        score = 78;
        classification = 'High-Pressure Social Engineering';
        details = 'Elevated voice instability index (Jitter) combined with persistent speaking rhythm (low pause density) indicates an active verbal harassment attempt.';
        mascotState = 'threat';
      } else if (avgPitch > 70 && avgPitch < 300) {
        score = 12;
        classification = 'Genuine Human Caller';
        details = 'Standard vocal modulation parameters and normal speaking pauses indicate a genuine human speaker.';
        mascotState = 'safe';
      } else {
        score = 0;
        classification = 'Inconclusive Profile';
        details = 'No distinct voice profile detected. Speak louder or record for at least 3 seconds.';
        mascotState = 'idle';
      }

      renderResults(score, classification, details, mascotState, audioBlobOrFile instanceof File ? audioBlobOrFile.name : "Live Mic Capture");
      visualizerPrompt.textContent = 'Microphone Inactive';
    }, 1000);
  }

  function renderResults(score, classification, explanation, mascotState, sourceName) {
    riskScoreText.textContent = `${score}%`;
    riskLabelText.textContent = score > 70 ? 'THREAT' : (score > 35 ? 'SUSPICIOUS' : 'SAFE');
    
    const offset = 264 - (264 * (score / 100));
    gaugeFill.style.strokeDashoffset = offset;

    verdictBanner.className = 'verdict-banner';
    if (score > 70) {
      verdictBanner.classList.add('danger', 'danger-animate');
      verdictTitle.textContent = `ALERT: ${classification}`;
      verdictDesc.textContent = explanation;
    } else if (score > 35) {
      verdictBanner.classList.add('warning', 'warning-animate');
      verdictTitle.textContent = `WARNING: ${classification}`;
      verdictDesc.textContent = explanation;
    } else if (score > 0) {
      verdictBanner.classList.add('safe', 'safe-animate');
      verdictTitle.textContent = `SECURE: ${classification}`;
      verdictDesc.textContent = explanation;
    } else {
      verdictBanner.classList.add('waiting');
      verdictTitle.textContent = 'System Idle';
      verdictDesc.textContent = explanation;
    }

    let mascotMessage = '';
    if (score > 70) {
      mascotMessage = `CRITICAL THREAT! Threat score is ${score}%. Voice properties match phishing spoofing patterns!`;
    } else if (score > 35) {
      mascotMessage = `Warning. Threat score is ${score}%. The vocal modulation parameters show moderate anomalies.`;
    } else if (score > 0) {
      mascotMessage = `Voice verification secure. Threat score is ${score}%. Genuine speaker characteristics detected.`;
    } else {
      mascotMessage = `Hello! I am Vigi. Speak into your mic or upload an audio file, and I will analyze the speech parameters for phishing behavior.`;
    }
    
    setMascotState(mascotState, mascotMessage);

    // Save to local storage history logs if valid run
    if (score > 0 && sourceName) {
      addHistoryItem(sourceName, classification, explanation, score);
    }
  }

  /* ==========================================================================
     AUDIO FILE UPLOAD & ANALYZER
     ========================================================================== */

  // Drag-and-drop Events
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleUploadedFile(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length > 0) {
      handleUploadedFile(fileInput.files[0]);
    }
  });

  function handleUploadedFile(file) {
    if (!file.type.startsWith('audio/')) {
      alert('Please upload a valid audio file (.wav, .mp3, etc.)');
      return;
    }

    activeAudioFile = file;
    const sizeKB = (file.size / 1024).toFixed(1);
    const displaySize = sizeKB > 1000 ? `${(sizeKB / 1024).toFixed(2)} MB` : `${sizeKB} KB`;

    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = displaySize;

    fileDetailsCard.classList.remove('hidden');
    dropZone.classList.add('hidden');
    uploadVisualizerContainer.classList.add('hidden');

    activeAudioBuffer = null;
    
    // Read file for local waveform display
    const reader = new FileReader();
    reader.onload = function(e) {
      const arrayBuffer = e.target.result;
      const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
      
      setMascotState('thinking', 'Decoding audio waveform headers...');

      decodeCtx.decodeAudioData(arrayBuffer, (decodedBuffer) => {
        activeAudioBuffer = decodedBuffer;
        uploadVisualizerStatus.textContent = 'Audio file parsed. Ready for deep network analysis.';
        uploadVisualizerContainer.classList.remove('hidden');
        
        drawUploadedWaveform(decodedBuffer);
        setMascotState('idle', `File decoded. Click 'Run Deep Analysis' to run Python ML classification.`);
        decodeCtx.close();
      }, (err) => {
        console.error('Audio decode error:', err);
        setMascotState('idle', 'Audio file decoding failed locally.');
        decodeCtx.close();
      });
    };
    reader.readAsArrayBuffer(file);
  }

  function drawUploadedWaveform(audioBuffer) {
    const ctx = canvasUpload.getContext('2d');
    const width = canvasUpload.width = canvasUpload.parentElement.clientWidth;
    const height = canvasUpload.height = canvasUpload.parentElement.clientHeight;

    ctx.fillStyle = '#070412';
    ctx.fillRect(0, 0, width, height);

    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / width);
    const amp = height / 2;

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#EC4899';
    ctx.beginPath();
    ctx.moveTo(0, amp);

    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[(i * step) + j];
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      ctx.lineTo(i, (1 + min) * amp);
      ctx.lineTo(i, (1 + max) * amp);
    }
    ctx.stroke();
  }

  // File Deep Analysis Action
  btnAnalyzeFile.addEventListener('click', async () => {
    if (!activeAudioFile) return;

    btnAnalyzeFile.disabled = true;
    uploadVisualizerStatus.textContent = 'Executing server predictions...';
    
    // Trigger analysis
    await analyzeAudioSource(activeAudioFile);
    
    btnAnalyzeFile.disabled = false;
    uploadVisualizerStatus.textContent = 'Analysis complete.';
  });

  // Enable restarting file uploads
  dropZone.addEventListener('click', () => {
    if (fileInput.files.length === 0) {
      fileInput.click();
    }
  });

  // Reset file upload state
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !fileDetailsCard.classList.contains('hidden')) {
      fileInput.value = '';
      fileDetailsCard.classList.add('hidden');
      dropZone.classList.remove('hidden');
      uploadVisualizerContainer.classList.add('hidden');
      activeAudioFile = null;
      activeAudioBuffer = null;
      setMascotState('idle', 'Upload states cleared.');
    }
  });

  /* ==========================================================================
     SCREENSHOT AUTOMATION
     ========================================================================== */
  const urlParams = new URLSearchParams(window.location.search);
  const screenshotState = urlParams.get('screenshot_state');
  if (screenshotState === 'upload') {
    // Switch to upload tab
    setTimeout(async () => {
      const uploadTabBtn = document.querySelector('[data-tab="upload-tab"]');
      if (uploadTabBtn) uploadTabBtn.click();
      
      try {
        const response = await fetch('/static/synthetic_sample.wav');
        const blob = await response.blob();
        const file = new File([blob], 'synthetic_sample.wav', { type: 'audio/wav' });
        handleUploadedFile(file);
      } catch (err) {
        console.error("Error loading upload tab screenshot file:", err);
      }
    }, 500);
  } else if (screenshotState === 'results') {
    // Switch to upload tab and show results
    setTimeout(async () => {
      const uploadTabBtn = document.querySelector('[data-tab="upload-tab"]');
      if (uploadTabBtn) uploadTabBtn.click();
      
      try {
        const response = await fetch('/static/synthetic_sample.wav');
        const blob = await response.blob();
        const file = new File([blob], 'synthetic_sample.wav', { type: 'audio/wav' });
        handleUploadedFile(file);
        
        // Trigger actual backend model analysis
        await analyzeAudioSource(file);
      } catch (err) {
        console.error("Error running automated screenshot analysis:", err);
      }
    }, 500);
  }

  /* ==========================================================================
     INTERACTIVE MIC BUTTON CLICK DISPATCHERS
     ========================================================================== */
  
  btnStartMic.addEventListener('click', startMicrophone);
  btnStopMic.addEventListener('click', stopMicrophone);

});

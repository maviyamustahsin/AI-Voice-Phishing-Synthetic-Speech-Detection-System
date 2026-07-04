# PROJECT GUIDE: AI-Based Voice Phishing (Vishing) Detection System
This guide provides a comprehensive technical overview of the project, details the algorithms, feature extraction pipeline, libraries, and design choices. Use this to prepare for examinations and presentation to the examiner.

---

## 🎓 1. Project Overview & Motivation
Traditional spam filters detect phishing by transcribing call speech to text (NLP) and searching for keyword patterns (e.g., "bank account", "password"). 

This approach has major flaws:
1. **Zero-Day Manipulation:** Fraudsters bypass keyword filters by using synonyms or speaking indirectly.
2. **Privacy Violations:** Transcribing user voice data onto a cloud server invades privacy.
3. **Deepfake Vulnerability:** It cannot detect if the voice itself is a computer-generated artificial deepfake.

**Our Approach:** We perform **non-semantic acoustic analysis**. Instead of analyzing *what* is said (words), we analyze *how* it is said (physical sound wave properties). This makes the system **language-agnostic, privacy-preserving, and highly robust** against synthetic voice cloning and high-pressure verbal tactics.

---

## 🛠️ 2. Core Libraries & Technology Stack

### Backend (Python)
- **Flask (v3.0.x) & Flask-CORS:** Sets up a local web server on `http://localhost:5000` to serve the website files and handle API requests.
- **Librosa (v0.10.x):** A high-performance audio analysis library. Used to load the uploaded audio files, compute frequencies, and perform mathematical transforms (Fourier transforms) on sound signals.
- **Scikit-Learn (v1.4.x):** Provides the Machine Learning pipeline, including the `RandomForestClassifier` and `StandardScaler` used to normalize acoustic features.
- **SoundFile & NumPy:** Handles binary decoding of audio buffers and mathematical array manipulations.
- **Pickle:** Serializes and loads the trained ML model (`vishing_model.pkl`) and normalization parameters (`vishing_scaler.pkl`).

### Frontend (HTML5 / Vanilla CSS / Modern JS)
- **Web Audio API:** Used inside `app.js` to capture live microphone input, create an analyzer node, and read raw amplitude bytes in real time.
- **HTML5 Canvas:** Dynamically renders the pulsing oscilloscope wave and visual frequencies based on real-time Web Audio API streams.
- **HTML5 LocalStorage:** Log history of tested audio files and statistics without needing a complex SQL database.

---

## 🧠 3. Machine Learning Dataset & Training

### The Dataset
The model is trained on the **Fake-or-Real (FoR) Dataset** (one of the largest deepfake audio detection datasets in research):
- Contains **195,000+ utterances**.
- Combines genuine human voices (from datasets like VoxForge, LJSpeech, and Arctic) with computer-generated synthetic speech.
- Includes speech synthesized by state-of-the-art Text-to-Speech (TTS) models (such as Google WaveNet, Tacotron, and Deep Voice 3).
- Incorporates telephone-line compression channels (rerec version) to mimic actual vishing call degradation.

### Training Details
- **Training Script:** `train_model.py` reads samples from `archive.zip`.
- **Sample Subset:** Evaluates random balanced samples (real vs. fake).
- **Features Extracted per Sample:** An 18-dimensional feature vector.
- **Validation Split:** 80% training / 20% validation.
- **Validation Accuracy:** **93.33%** achieved using the Random Forest algorithm.

---

## 🎛️ 4. Feature Extraction Pipeline (18 Dimensions)
For every audio snippet analyzed, the system extracts exactly **18 features** to create the input vector for the machine learning classifier:

| Feature Name | Dimension | Description & VIVA Explanation |
| :--- | :--- | :--- |
| **Mel-Frequency Cepstral Coefficients (MFCCs)** | 13 features (MFCC 1-13) | Translates raw audio signals into the **Mel Scale** (which mimics the logarithmic pitch perception of the human ear). It captures the spectral envelope (timbre, vocal tract shape), which is a key signature to distinguish human speakers from vocoder/synthetic sound systems. |
| **Pitch (Fundamental Frequency - $F_0$)** | 1 feature | The base frequency of vocal cords vibration. Calculated using autocorrelation. Standard human range is 85–180 Hz (Male) and 165–255 Hz (Female). Anomalous pitch behavior can indicate robot voice boxes. |
| **Pitch Stability (Jitter)** | 1 feature | Measures pitch instability. Natural human voices exhibit slight modulation (organic jitter). Synthetic voices are often extremely flat/monotone (very low jitter). High-pressure scammers show nervous micro-tremors (high jitter). |
| **Spectral Centroid** | 1 feature | Indicates the "brightness" center of gravity of the frequency spectrum. Synthetic/robotic voices contain artificial high-frequency vocoder artifacts, resulting in an elevated, metallic Spectral Centroid ($> 3000\text{ Hz}$). |
| **Root Mean Square (RMS) Energy** | 1 feature | Measures signal amplitude/volume. Used to determine the strength of the voice signal and calculate pause boundaries. |
| **Zero-Crossing Rate (ZCR)** | 1 feature | The rate at which the sound wave crosses the zero amplitude line. Helps separate voiced/vowel sounds (low ZCR) from unvoiced consonants (high ZCR) and digital clipping noise. |

*Note: The pause ratio (speech rhythm) is calculated based on volume drops and combined with the ML output to flag high-pressure social engineering calls.*

---

## 🤖 5. Machine Learning Algorithm
The project implements a **Random Forest Classifier**:
- **Why Random Forest?**
  1. **Non-Linear Decision Boundaries:** Acoustic features (like pitch and spectral shape) interact in complex, non-linear ways. Simple linear models (Logistic Regression) perform poorly.
  2. **Reduced Overfitting:** It builds an ensemble (forest) of 100 individual decision trees and averages their votes, making it highly robust to noise.
  3. **Low Latency:** Inference (classification) happens in under 5 milliseconds, enabling real-time detection on localhost.
- **Feature Standardization:** Features are scaled using `StandardScaler` so that features with large numerical ranges (like Spectral Centroid in thousands of Hz) do not dominate smaller values (like RMS energy).

---

## 🔑 6. API Keys (Why there are NONE)
A major selling point of this project to the examiner is the **Zero External API Dependency**:
1. **Privacy Protection:** No voice recordings are uploaded to external clouds (e.g., OpenAI, Google, AWS). The audio is processed purely in local RAM and temporary workspace.
2. **Offline-Ready:** Works without an active internet connection.
3. **No Fees:** Free to run without API tokens or subscription limits.

---

## 🔄 7. Step-by-Step Execution Flow
When a user launches the app, the data flows as follows:

```mermaid
graph TD
    A[User clicks Run_PhishGuard_AI.bat] -->|1. Starts Flask Server| B(server.py)
    A -->|2. Opens Web Browser| C[http://localhost:5000]
    C -->|3. Captures Voice input| D[Live Mic / File Upload]
    D -->|4. Sends POST request to API| E[/api/analyze]
    E -->|5. Runs Python script| F(feature_extractor.py)
    F -->|6. Extracts 18 acoustic metrics| G[Feature Vector]
    G -->|7. Scales & predicts| H(vishing_model.pkl)
    H -->|8. Returns verdict details| E
    E -->|9. Renders result visually| C
    C -->|10. Saves log record| I[localStorage History]
```

1. **Launcher Activation:** Double-clicking `Run_PhishGuard_AI.bat` starts `server.py` in the background and opens the web browser automatically.
2. **Audio Input:** The user records speech or uploads an audio file.
3. **API Payload:** The audio is sent as a `multipart/form-data` payload via a `POST` request to `/api/analyze`.
4. **Feature Extraction:** Python's `librosa` processes the audio in memory and extracts Jitter, Centroid, and MFCCs.
5. **Model Evaluation:** The vector is standardized by the scaler and processed by the Random Forest model.
6. **Verdict Return:** Flask returns a JSON response containing the threat score (0-100), classification verdict, detailed explanation, and mascot emotion state.
7. **Animation and Log:** The frontend renders the gauge, applies glowing CSS animations, and saves the result to the browser's `localStorage` database, instantly updating the History table.

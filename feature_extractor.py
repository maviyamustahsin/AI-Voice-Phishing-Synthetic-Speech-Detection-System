import librosa
import numpy as np

def extract_features_from_audio(y, sr):
    """
    Extracts advanced acoustic features from an audio time-series signal.
    Returns:
        features_vector: 24-element numpy array for model training/classification.
        metrics_dict: Dictionary of human-readable feature metrics.
    """
    # Ensure minimum audio length
    if len(y) < 1000:
        return None, None
        
    # 1. MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # 2. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    
    # 3. RMS Energy (Volume)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    
    # 4. Zero Crossing Rate (Spectral flatness/high frequency noise)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.mean(zcr)
    
    # 5. Advanced: Chroma STFT (Select 4 dimensions to represent harmonic profile)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma.T, axis=0)
    # Take 4 principal bins/averages to represent pitch-class profiles
    chroma_4 = np.array([
        np.mean(chroma_mean[0:3]), # Bass / low register harmonics
        np.mean(chroma_mean[3:6]), # Low-mid register harmonics
        np.mean(chroma_mean[6:9]), # Mid-high register harmonics
        np.mean(chroma_mean[9:12]) # High register harmonics
    ])

    # 6. Advanced: Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_mean = np.mean(rolloff)

    # 7. Advanced: Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)

    # 8. Pitch tracking (F0 estimation using YIN)
    try:
        # We specify fmin=60 and fmax=400 to match typical human voices
        f0 = librosa.yin(y=y, sr=sr, fmin=60, fmax=400)
        # Clean infinite/NaN readings
        f0 = f0[~np.isnan(f0)]
        f0 = f0[~np.isinf(f0)]
        f0 = f0[f0 > 0]
        if len(f0) > 0:
            pitch_mean = np.mean(f0)
            pitch_std = np.std(f0)
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
    except Exception:
        pitch_mean = 0.0
        pitch_std = 0.0

    # Combine features into an 24-dimensional vector
    features_vector = np.hstack([
        mfccs_mean,       # 13
        pitch_mean,       # 1
        pitch_std,        # 1
        centroid_mean,    # 1
        rms_mean,         # 1
        zcr_mean,         # 1
        chroma_4,         # 4
        rolloff_mean,     # 1
        bandwidth_mean    # 1
    ])
    
    metrics_dict = {
        "pitch": float(pitch_mean),
        "jitter": float(pitch_std),
        "centroid": float(centroid_mean),
        "rms": float(rms_mean),
        "zcr": float(zcr_mean),
        "rolloff": float(rolloff_mean),
        "bandwidth": float(bandwidth_mean),
        "chroma_bass": float(chroma_4[0]),
        "chroma_mid": float(chroma_4[1]),
        "chroma_treble": float(chroma_4[2])
    }
    
    return features_vector, metrics_dict

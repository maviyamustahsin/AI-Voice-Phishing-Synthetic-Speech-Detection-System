import zipfile
import io
import os
import pickle
import json
import time
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from feature_extractor import extract_features_from_audio

ZIP_PATH = r"c:\Users\Administrator\Desktop\project\archive.zip"
MODEL_PATH = r"c:\Users\Administrator\Desktop\project\vishing_model.pkl"
SCALER_PATH = r"c:\Users\Administrator\Desktop\project\vishing_scaler.pkl"
BENCHMARK_PATH = r"c:\Users\Administrator\Desktop\project\models_benchmark.json"

def main():
    if not os.path.exists(ZIP_PATH):
        print(f"Error: Dataset zip file not found at {ZIP_PATH}")
        return

    print("Opening archive.zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        namelist = zip_ref.namelist()
        
        # Gather file paths
        real_files = [n for n in namelist if n.startswith("for-2sec/") and "/training/real/" in n and n.endswith(".wav")]
        fake_files = [n for n in namelist if n.startswith("for-2sec/") and "/training/fake/" in n and n.endswith(".wav")]
        
        print(f"Total training real files found in zip: {len(real_files)}")
        print(f"Total training fake files found in zip: {len(fake_files)}")
        
        # Sample subset to train quickly (engineering presentation scale)
        num_samples = 300
        sampled_real = np.random.choice(real_files, num_samples, replace=False)
        sampled_fake = np.random.choice(fake_files, num_samples, replace=False)
        
        X = []
        y = []
        
        # Process Real Files (Label 0)
        print("\nExtracting features (24 dimensions) from GENUINE human samples...")
        success_count = 0
        for i, file_path in enumerate(sampled_real):
            if i % 100 == 0:
                print(f"  Processing real file {i}/{num_samples}...")
            try:
                with zip_ref.open(file_path) as f:
                    audio_data, sample_rate = sf.read(io.BytesIO(f.read()))
                features, _ = extract_features_from_audio(audio_data, sample_rate)
                if features is not None:
                    X.append(features)
                    y.append(0) # 0 = Genuine
                    success_count += 1
            except Exception as e:
                continue
        print(f"Successfully processed {success_count} genuine samples.")

        # Process Fake Files (Label 1)
        print("\nExtracting features (24 dimensions) from SYNTHETIC machine samples...")
        success_count_fake = 0
        for i, file_path in enumerate(sampled_fake):
            if i % 100 == 0:
                print(f"  Processing synthetic file {i}/{num_samples}...")
            try:
                with zip_ref.open(file_path) as f:
                    audio_data, sample_rate = sf.read(io.BytesIO(f.read()))
                features, _ = extract_features_from_audio(audio_data, sample_rate)
                if features is not None:
                    X.append(features)
                    y.append(1) # 1 = Phishing/Synthetic
                    success_count_fake += 1
            except Exception as e:
                continue
        print(f"Successfully processed {success_count_fake} synthetic samples.")

        X = np.array(X)
        y = np.array(y)
        print(f"\nTotal dataset shape: {X.shape}")
        
        # Split into train & validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Save scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"StandardScaler saved to: {SCALER_PATH}")

        # Models to evaluate
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
            "Support Vector Machine": SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
            "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
        }

        benchmark_results = {}
        best_model_name = ""
        best_val_acc = 0.0
        best_model_object = None

        print("\n" + "="*80)
        print(f"{'Algorithm Comparison Matrix (24 Advanced Features)':^80}")
        print("="*80)
        print(f"{'Classifier':<25} | {'Val Acc':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Latency':<8}")
        print("-"*80)

        for name, clf in models.items():
            # Training
            t0 = time.time()
            clf.fit(X_train_scaled, y_train)
            train_time = time.time() - t0
            
            # Predict & Evaluate
            y_pred = clf.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, y_pred) * 100
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
            
            # Measure inference latency (single sample average over 100 runs)
            t_inf_start = time.time()
            for _ in range(100):
                clf.predict([X_val_scaled[0]])
            latency_ms = ((time.time() - t_inf_start) / 100) * 1000

            print(f"{name:<25} | {val_acc:>7.2f}% | {precision:>9.2f} | {recall:>9.2f} | {f1:>9.2f} | {latency_ms:>6.3f}ms")
            
            benchmark_results[name] = {
                "val_accuracy": round(val_acc, 2),
                "precision": round(float(precision), 3),
                "recall": round(float(recall), 3),
                "f1_score": round(float(f1), 3),
                "latency_ms": round(latency_ms, 3),
                "train_time_sec": round(train_time, 3)
            }

            # Pick best model based on F1/Accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_name = name
                best_model_object = clf

        print("="*80)
        print(f"Selected Best Classifier: {best_model_name} ({best_val_acc:.2f}% validation accuracy)")
        
        # Save best model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(best_model_object, f)
        print(f"Best classifier saved to: {MODEL_PATH}")

        # Save comparative stats as JSON for Flask to serve
        with open(BENCHMARK_PATH, 'w') as f:
            json.dump({
                "selected_best": best_model_name,
                "matrix": benchmark_results
            }, f, indent=2)
        print(f"Benchmark results JSON saved to: {BENCHMARK_PATH}")

if __name__ == '__main__':
    main()

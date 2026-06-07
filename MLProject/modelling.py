import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def main():
    print("[INFO] Memulai eksekusi Pipeline CI/CD dengan Autolog...")
    
    # 1. Memuat Data
    if not os.path.exists('dataset_fintech_clean.csv'):
        raise FileNotFoundError("dataset_fintech_clean.csv tidak ditemukan di direktori eksekusi!")
        
    df = pd.read_csv('dataset_fintech_clean.csv').dropna()
    X = df['teks_bersih']
    y = df['sentimen']

    # 2. Ekstraksi Fitur
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # 3. AKTIFKAN AUTOMATIC LOGGING (Wajib Kriteria Basic)
    mlflow.autolog()

    # 4. Eksekusi Training di dalam Run
    with mlflow.start_run() as run:
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=2, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"[INFO] Akurasi Model: {acc:.4f} | F1-Score: {f1:.4f}")

        # Simpan TF-IDF Vectorizer secara manual (karena autolog hanya mencatat komponen sklearn)
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact("tfidf_vectorizer.pkl", "preprocessing")
        
        # Simpan RUN_ID ke file teks untuk kebutuhan GitHub Actions Docker Build
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

    print("[SUKSES] Model CI/CD telah dilatih dengan autolog dan direkam ke MLflow.")

if __name__ == "__main__":
    main()

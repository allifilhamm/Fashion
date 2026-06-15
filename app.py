import pandas as pd
import pickle
import numpy as np
from surprise import accuracy, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

print("Loading data dan model...")

df = pd.read_csv("fashion_dataset_final.csv")

cf_model  = pickle.load(open("cf_model.pkl",  "rb"))
svd_model = pickle.load(open("svd_model.pkl", "rb"))
knn_model = pickle.load(open("knn_model.pkl", "rb"))

reader = Reader(rating_scale=(df["Rating"].min(), df["Rating"].max()))

data = Dataset.load_from_df(
    df[["User_ID", "Clothing ID", "Rating"]],
    reader
)

# Gunakan split yang sama (random_state=42) supaya konsisten
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

def evaluate_model(model, testset, threshold=3.5):
    """
    Hitung RMSE, MAE, Precision, Recall, F1 dari model Surprise.
    threshold: rating >= threshold dianggap 'relevan' (positif)
    """

    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)

    # Binarisasi untuk Precision / Recall / F1
    y_true = [1 if pred.r_ui >= threshold else 0 for pred in predictions]
    y_pred = [1 if pred.est  >= threshold else 0 for pred in predictions]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    return {
        "RMSE":      round(rmse,      4),
        "MAE":       round(mae,       4),
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "F1-Score":  round(f1,        4),
    }

print("Mengevaluasi CF...")
cf_metrics  = evaluate_model(cf_model,  testset)

print("Mengevaluasi SVD...")
svd_metrics = evaluate_model(svd_model, testset)

print("Mengevaluasi KNN...")
knn_metrics = evaluate_model(knn_model, testset)

result_df = pd.DataFrame([
    {"Model": "CF",  **cf_metrics},
    {"Model": "SVD", **svd_metrics},
    {"Model": "KNN", **knn_metrics},
])

print("\n=== Hasil Evaluasi ===")
print(result_df.to_string(index=False))

result_df.to_excel("hasil_evaluasi.xlsx", index=False)

print("\nhasil_evaluasi.xlsx berhasil diperbarui!")

Python
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np  
from surprise import Reader, Dataset, accuracy  
from surprise.model_selection import train_test_split  
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    mean_squared_error, 
    mean_absolute_error
)

st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👗",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("fashion_dataset_final.csv")

@st.cache_resource
def load_models():
    cf = pickle.load(open("cf_model.pkl", "rb"))
    svd = pickle.load(open("svd_model.pkl", "rb"))
    knn = pickle.load(open("knn_model.pkl", "rb"))
    return cf, svd, knn

df = load_data()
cf_model, svd_model, knn_model = load_models()


if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.title("👗 Fashion Recommendation System")

    st.markdown("### Login Sistem")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "admin123":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Username atau password salah")

    st.stop()

st.sidebar.title("Fashion Recommendation")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Dashboard",
        "Katalog Produk",
        "Histori User",
        "Rekomendasi Produk",
        "Visualisasi Akurasi"
    ]
)


def get_top_n_recommendations(
    user_id,
    model,
    df,
    n=10
):

    all_products = df["Clothing ID"].unique()

    rated_products = df[
        df["User_ID"] == user_id
    ]["Clothing ID"].unique()

    unseen_products = [
        product
        for product in all_products
        if product not in rated_products
    ]

    predictions = []

    for product in unseen_products:

        pred = model.predict(
            uid=user_id,
            iid=product
        )

        predictions.append(
            (
                product,
                pred.est
            )
        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_n = predictions[:n]

    recommendation_df = pd.DataFrame(
        top_n,
        columns=[
            "Clothing_ID",
            "Predicted_Rating"
        ]
    )

    product_info = df[
        [
            "Clothing ID",
            "Class Name",
            "Department Name"
        ]
    ].drop_duplicates()

    result = recommendation_df.merge(
        product_info,
        left_on="Clothing_ID",
        right_on="Clothing ID",
        how="left"
    )

    return result[
        [
            "Clothing_ID",
            "Class Name",
            "Department Name",
            "Predicted_Rating"
        ]
    ]

@st.cache_data
def build_user_item_matrix(_df):
    """
    Buat user-item matrix dari dataframe.
    Baris = user, kolom = produk.
    """
    matrix = _df.pivot_table(
        index="User_ID",
        columns="Clothing ID",
        values="Rating",
        aggfunc="mean"
    ).fillna(0)
    return matrix

def get_top_n_recommendations_knn(
    user_id,
    model,
    df,
    n=10
):
    """
    Rekomendasi dengan sklearn NearestNeighbors (user-based).
    Langkah:
    1. Cari K user paling mirip dengan user_id
    2. Ambil produk yang disukai user-user tetangga
    3. Filter produk yang belum pernah dilihat user_id
    4. Urutkan berdasarkan rata-rata rating tetangga
    """

    matrix = build_user_item_matrix(df)

    if hasattr(model, 'feature_names_in_'):
        matrix = matrix.reindex(columns=model.feature_names_in_, fill_value=0)
   
    if user_id not in matrix.index:
        return pd.DataFrame(
            columns=[
                "Clothing_ID",
                "Class Name",
                "Department Name",
                "Predicted_Rating"
            ]
        )

    user_idx = matrix.index.get_loc(user_id)
    user_vector = matrix.iloc[user_idx].values.reshape(1, -1)

    k = min(20, model.n_samples_fit_)
    distances, indices = model.kneighbors(user_vector, n_neighbors=k)

    neighbor_indices = indices[0]
    neighbor_distances = distances[0]

    rated_products = set(
        df[df["User_ID"] == user_id]["Clothing ID"].unique()
    )

    product_scores = {}

    for neighbor_idx, distance in zip(neighbor_indices, neighbor_distances):

        neighbor_user_id = matrix.index[neighbor_idx]
        similarity = 1 / (1 + distance)  # Ubah distance → similarity

        neighbor_ratings = df[
            df["User_ID"] == neighbor_user_id
        ][["Clothing ID", "Rating"]]

        for _, row in neighbor_ratings.iterrows():

            product = row["Clothing ID"]
            rating = row["Rating"]

            if product in rated_products:
                continue

            if product not in product_scores:
                product_scores[product] = {"score": 0, "weight": 0}

            product_scores[product]["score"]  += similarity * rating
            product_scores[product]["weight"] += similarity

    predictions = []

    for product, val in product_scores.items():
        if val["weight"] > 0:
            predicted_rating = val["score"] / val["weight"]
            predictions.append((product, round(predicted_rating, 4)))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    if not top_n:
        return pd.DataFrame(
            columns=[
                "Clothing_ID",
                "Class Name",
                "Department Name",
                "Predicted_Rating"
            ]
        )

    recommendation_df = pd.DataFrame(
        top_n,
        columns=["Clothing_ID", "Predicted_Rating"]
    )

    product_info = df[
        ["Clothing ID", "Class Name", "Department Name"]
    ].drop_duplicates()

    result = recommendation_df.merge(
        product_info,
        left_on="Clothing_ID",
        right_on="Clothing ID",
        how="left"
    )

    return result[
        ["Clothing_ID", "Class Name", "Department Name", "Predicted_Rating"]
    ]

if menu == "Dashboard":

    st.title("📊 Dashboard Admin")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total User",
        df["User_ID"].nunique()
    )

    col2.metric(
        "Total Produk",
        df["Clothing ID"].nunique()
    )

    col3.metric(
        "Total Interaksi",
        len(df)
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Distribusi Rating")

        fig, ax = plt.subplots()

        df["Rating"].value_counts().sort_index().plot(
            kind="bar",
            ax=ax
        )

        plt.xlabel("Rating")
        plt.ylabel("Jumlah")

        st.pyplot(fig)

    with right:

        st.subheader("Top Kategori Produk")

        fig2, ax2 = plt.subplots()

        df["Class Name"].value_counts().head(10).plot(
            kind="bar",
            ax=ax2
        )

        st.pyplot(fig2)

elif menu == "Katalog Produk":

    st.title("🛍️ Katalog Produk")

    produk = df[
        [
            "Clothing ID",
            "Class Name",
            "Department Name"
        ]
    ].drop_duplicates()

    search = st.text_input(
        "Cari Produk"
    )

    if search:

        produk = produk[
            produk["Class Name"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.dataframe(
        produk,
        use_container_width=True
    )


elif menu == "Histori User":

    st.title("👤 Histori User")

    user_id = st.number_input(
        "Masukkan User ID",
        min_value=1,
        value=1,
        step=1
    )

    user_id = int(user_id)

    history = df[
        df["User_ID"] == user_id
    ]

    st.write(
        f"Jumlah Interaksi: {len(history)}"
    )

    st.dataframe(
        history[
            [
                "Clothing ID",
                "Class Name",
                "Rating"
            ]
        ],
        use_container_width=True
    )

elif menu == "Rekomendasi Produk":

    st.title("🎯 Sistem Rekomendasi Produk")

    user_id = st.number_input(
        "Masukkan User ID",
        min_value=1,
        value=1,
        step=1
    )

    user_id = int(user_id)

    model_choice = st.selectbox(
        "Pilih Model",
        [
            "SVD",
            "Collaborative Filtering",
            "KNN"
        ]
    )

    if model_choice == "SVD":
        selected_model = svd_model
        use_knn = False
    elif model_choice == "KNN":
        selected_model = knn_model
        use_knn = True
    else:
        selected_model = cf_model
        use_knn = False

    if st.button("Generate Recommendation"):

        with st.spinner("Membuat rekomendasi..."):

            if use_knn:
                result = get_top_n_recommendations_knn(
                    user_id=user_id,
                    model=selected_model,
                    df=df,
                    n=10
                )
            else:
                result = get_top_n_recommendations(
                    user_id=user_id,
                    model=selected_model,
                    df=df,
                    n=10
                )

        if result.empty:
            st.warning(
                "Tidak ada rekomendasi yang bisa dibuat untuk user ini dengan model KNN. "
                "Pastikan user memiliki cukup histori interaksi."
            )
        else:
            st.success(
                f"Rekomendasi berhasil dibuat menggunakan model {model_choice}"
            )

            st.dataframe(
                result,
                use_container_width=True
            )

            csv = result.to_csv(
                index=False
            ).encode()

            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"recommendation_{model_choice.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

reader = Reader(rating_scale=(df["Rating"].min(), df["Rating"].max()))
 
data = Dataset.load_from_df(
    df[["User_ID", "Clothing ID", "Rating"]],
    reader
)
 
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
 
def evaluate_surprise_model(model, testset, threshold=3.5):
    predictions = model.test(testset)
 
    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)
 
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

 
def evaluate_knn_model(knn_model, df, threshold=3.5, test_size=0.2, random_state=42):
    """
    Evaluasi sklearn NearestNeighbors (user-based collaborative filtering).
    - Buat user-item matrix
    - Split user secara acak jadi train/test
    - Untuk setiap test user, prediksi rating produk yang belum dilihat
      berdasarkan weighted average rating dari K tetangga terdekat
    - Hitung RMSE, MAE, Precision, Recall, F1
    """
 
    print("  Membuat user-item matrix...")
 
    matrix = df.pivot_table(
        index="User_ID",
        columns="Clothing ID",
        values="Rating",
        aggfunc="mean"
    ).fillna(0)
 
    if hasattr(knn_model, 'feature_names_in_'):
        matrix = matrix.reindex(columns=knn_model.feature_names_in_, fill_value=0)

    all_users = matrix.index.tolist()
 
    np.random.seed(random_state)
    test_users = np.random.choice(
        all_users,
        size=int(len(all_users) * test_size),
        replace=False
    )
 
    y_true_all = []
    y_pred_all = []
 
    print(f"  Mengevaluasi {len(test_users)} test users...")
 
    for i, user_id in enumerate(test_users):
 
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(test_users)}")
 
        user_idx = matrix.index.get_loc(user_id)
        user_vector = matrix.iloc[user_idx].values.reshape(1, -1)
 
        k = min(20, knn_model.n_samples_fit_)
        distances, indices = knn_model.kneighbors(user_vector, n_neighbors=k)
 
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
        user_ratings = df[df["User_ID"] == user_id][["Clothing ID", "Rating"]]
 
        if user_ratings.empty:
            continue
        product_scores = {}
 
        for neighbor_idx, distance in zip(neighbor_indices, neighbor_distances):
 
            neighbor_user_id = matrix.index[neighbor_idx]
            if neighbor_user_id == user_id:
                continue
 
            similarity = 1 / (1 + distance)
 
            neighbor_ratings = df[
                df["User_ID"] == neighbor_user_id
            ][["Clothing ID", "Rating"]]
 
            for _, row in neighbor_ratings.iterrows():
                product = row["Clothing ID"]
                rating  = row["Rating"]
 
                if product not in product_scores:
                    product_scores[product] = {"score": 0, "weight": 0}
 
                product_scores[product]["score"]  += similarity * rating
                product_scores[product]["weight"] += similarity
 
        for _, row in user_ratings.iterrows():
            product = row["Clothing ID"]
            actual  = row["Rating"]
 
            if product in product_scores and product_scores[product]["weight"] > 0:
                predicted = product_scores[product]["score"] / product_scores[product]["weight"]
                y_true_all.append(actual)
                y_pred_all.append(predicted)
 
    if not y_true_all:
        print("  Tidak ada prediksi yang bisa dievaluasi!")
        return {
            "RMSE": None, "MAE": None,
            "Precision": None, "Recall": None, "F1-Score": None
        }
 
    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
 
    rmse = round(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)), 4)
    mae  = round(mean_absolute_error(y_true_arr, y_pred_arr), 4)
 
    y_true_bin = [1 if v >= threshold else 0 for v in y_true_arr]
    y_pred_bin = [1 if v >= threshold else 0 for v in y_pred_arr]
 
    precision = round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 4)
    recall    = round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 4)
    f1        = round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 4)
 
    return {
        "RMSE": rmse, "MAE": mae,
        "Precision": precision, "Recall": recall, "F1-Score": f1
    }

 
print("\nMengevaluasi CF...")
cf_metrics  = evaluate_surprise_model(cf_model,  testset)
 
print("Mengevaluasi SVD...")
svd_metrics = evaluate_surprise_model(svd_model, testset)
 
print("Mengevaluasi KNN (sklearn)...")
knn_metrics = evaluate_knn_model(knn_model, df)

 
result_df = pd.DataFrame([
    {"Model": "CF",  **cf_metrics},
    {"Model": "SVD", **svd_metrics},
    {"Model": "KNN", **knn_metrics},
])
 
print("\n=== Hasil Evaluasi ===")
print(result_df.to_string(index=False))
 
result_df.to_excel("hasil_evaluasi.xlsx", index=False)
print("\nhasil_evaluasi.xlsx berhasil diperbarui!")
 

st.sidebar.divider()

st.sidebar.info(
    """
    Fashion Recommendation System

    Collaborative Filtering
    KNN
    SVD

    CRISP-DM Methodology
    """
)

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO

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


def get_top_n_recommendations_knn(
    user_id,
    model,
    df,
    n=10
):
    """
    KNN dari Surprise (KNNBasic / KNNWithMeans / KNNWithZScore / KNNBaseline)
    menggunakan model.predict() dengan interface yang sama seperti SVD,
    sehingga cukup panggil fungsi ini dengan cara yang sama.
    """

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

        try:
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
        except Exception:
            # Lewati produk yang tidak bisa diprediksi
            continue

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

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

elif menu == "Visualisasi Akurasi":

    st.title("📈 Evaluasi Model")

    try:

        result_df = pd.read_excel(
            "hasil_evaluasi.xlsx"
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots()

            ax.bar(
                result_df["Model"],
                result_df["RMSE"]
            )

            ax.set_title(
                "Perbandingan RMSE"
            )

            st.pyplot(fig)

        with col2:

            fig2, ax2 = plt.subplots()

            ax2.bar(
                result_df["Model"],
                result_df["MAE"]
            )

            ax2.set_title(
                "Perbandingan MAE"
            )

            st.pyplot(fig2)

    except:

        st.warning(
            "hasil_evaluasi.xlsx belum ditemukan"
        )

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

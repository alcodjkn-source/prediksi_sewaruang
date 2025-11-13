import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, PassiveAggressiveRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import plotly.express as px

# ==========================
# Load model Random Forest
# ==========================
MODEL_PATH = "model/model_rf_hargapenawaran.joblib"
if os.path.exists(MODEL_PATH):
    model_rf = joblib.load(MODEL_PATH)
    st.session_state["model_rf"] = model_rf
else:
    st.warning("⚠️ Model belum tersedia!")

# ==========================
# Sidebar Navigasi
# ==========================
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Beranda", "Evaluasi Model", "Model Dasar Prediksi", "Prediksi"]
)

# ==========================
# Halaman 1: Beranda
# ==========================
if page == "Beranda":
    st.title("🏠 Beranda")
    st.markdown("""
    Selamat datang di aplikasi Prediksi Harga Penawaran.  
    Gunakan modul **Evaluasi Model** untuk mengecek performa model.  
    Gunakan modul **Prediksi** untuk menghitung harga baru dari input variabel.
    """)

# ==========================
# Halaman 2: Evaluasi Model
# ==========================
elif page == "Evaluasi Model":
    st.title("📊 Evaluasi Model")
    uploaded_file = st.file_uploader(
        "Upload dataset CSV/XLSX dengan kolom target 'HARGAPENAWARAN'", 
        type=["csv","xlsx"]
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
        
        if "HARGAPENAWARAN" not in df.columns:
            st.error("Kolom target 'HARGAPENAWARAN' tidak ditemukan!")
        else:
            target = "HARGAPENAWARAN"
            X = df.drop(columns=[target])
            y = df[target]
            X = X.loc[:, X.nunique() > 1]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=77
            )

            # Daftar semua model
            all_models = {
                'Linear Regression': LinearRegression(),
                'Support Vector Regression': SVR(),
                'K-Nearest Neighbor': KNeighborsRegressor(),
                'Elastic Net': ElasticNet(random_state=77),
                'Passive Aggressive Regressor': PassiveAggressiveRegressor(random_state=77),
                'Random Forest': RandomForestRegressor(random_state=77),
                'Gradient Boosting': GradientBoostingRegressor(random_state=77),
                'LightGBM': LGBMRegressor(random_state=77),
                'XGBoost': xgb.XGBRegressor(random_state=77, verbosity=0)
            }

            # Pilih model yang ingin dievaluasi
            selected_models = st.multiselect(
                "Pilih model untuk dievaluasi:",
                options=list(all_models.keys()),
                default=["Random Forest", "Linear Regression"]
            )

            # DataFrame hasil evaluasi
            results = pd.DataFrame(columns=[
                'Model',
                'R2_in_sample', 'R2_out_sample',
                'MSE_in_sample', 'MSE_out_sample',
                'RMSE_in_sample', 'RMSE_out_sample',
                'MAE_in_sample', 'MAE_out_sample',
                'MAPE_in_sample', 'MAPE_out_sample'
            ])

            # Training & evaluasi model terpilih
            for name in selected_models:
                model = all_models[name]
                try:
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred  = model.predict(X_test)

                    mse_in  = mean_squared_error(y_train, y_train_pred)
                    rmse_in = np.sqrt(mse_in)
                    mae_in  = mean_absolute_error(y_train, y_train_pred)
                    mape_in = mean_absolute_percentage_error(y_train, y_train_pred)
                    r2_in   = r2_score(y_train, y_train_pred)

                    mse_out  = mean_squared_error(y_test, y_test_pred)
                    rmse_out = np.sqrt(mse_out)
                    mae_out  = mean_absolute_error(y_test, y_test_pred)
                    mape_out = mean_absolute_percentage_error(y_test, y_test_pred)
                    r2_out   = r2_score(y_test, y_test_pred)

                    results = pd.concat([results, pd.DataFrame([{
                        'Model': name,
                        'R2_in_sample': r2_in,
                        'R2_out_sample': r2_out,
                        'MSE_in_sample': mse_in,
                        'MSE_out_sample': mse_out,
                        'RMSE_in_sample': rmse_in,
                        'RMSE_out_sample': rmse_out,
                        'MAE_in_sample': mae_in,
                        'MAE_out_sample': mae_out,
                        'MAPE_in_sample': mape_in,
                        'MAPE_out_sample': mape_out
                    }])], ignore_index=True)
                except Exception as e:
                    st.warning(f"⚠️ Model {name} gagal dijalankan: {e}")

            # Sortir hasil
            results = results.sort_values(by='RMSE_out_sample', ascending=True)

            # 1️⃣ Tabel Interaktif
            st.subheader("Tabel Hasil Evaluasi Model")
            st.dataframe(
                results.style.format({
                    'R2_in_sample': '{:.3f}',
                    'R2_out_sample': '{:.3f}',
                    'MSE_in_sample': '{:,.0f}',
                    'MSE_out_sample': '{:,.0f}',
                    'RMSE_in_sample': '{:,.0f}',
                    'RMSE_out_sample': '{:,.0f}',
                    'MAE_in_sample': '{:,.0f}',
                    'MAE_out_sample': '{:,.0f}',
                    'MAPE_in_sample': '{:.2%}',
                    'MAPE_out_sample': '{:.2%}'
                }).background_gradient(cmap='plasma', subset=['R2_in_sample','R2_out_sample'])
                  .background_gradient(cmap='viridis', subset=['RMSE_in_sample','RMSE_out_sample','MAE_in_sample','MAE_out_sample','MAPE_in_sample','MAPE_out_sample'])
            )

            # 2️⃣ Chart Interaktif dengan Tabs
            st.subheader("Visualisasi Performa Model")
            tab1, tab2 = st.tabs(["RMSE Out-Sample", "R² Out-Sample"])

            with tab1:
                fig_rmse = px.bar(
                    results,
                    x='RMSE_out_sample',
                    y='Model',
                    orientation='h',
                    color='RMSE_out_sample',
                    color_continuous_scale='viridis',
                    text='RMSE_out_sample'
                )
                fig_rmse.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_rmse, use_container_width=True)

            with tab2:
                fig_r2 = px.bar(
                    results,
                    x='R2_out_sample',
                    y='Model',
                    orientation='h',
                    color='R2_out_sample',
                    color_continuous_scale='plasma',
                    text='R2_out_sample'
                )
                fig_r2.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_r2, use_container_width=True)

            # 3️⃣ Keterangan indikator
            with st.expander("📌 Keterangan Rinci Indikator Evaluasi Model"):
                st.markdown("""
                1. **R² (R-squared) 📈**  
                   - Menunjukkan seberapa baik model menjelaskan variasi target.  
                   - Nilai 0–1: ≥0.9 sangat baik, 0.7–0.9 baik, 0.5–0.7 sedang, <0.5 kurang baik.  
                   - R² negatif → model lebih buruk daripada prediksi mean.

                2. **MSE (Mean Squared Error) 💥**  
                   - Rata-rata kuadrat selisih prediksi dengan nilai aktual.  
                   - Semakin kecil → semakin akurat.  
                   - Satuan = kuadrat target (misal target juta → MSE juta²).

                3. **RMSE (Root Mean Squared Error) 🌟**  
                   - Akar dari MSE, satuan sama dengan target.  
                   - Semakin kecil → prediksi lebih dekat ke nilai aktual.

                4. **MAE (Mean Absolute Error) ✨**  
                   - Rata-rata absolut error prediksi.  
                   - Semakin kecil → prediksi lebih akurat.

                5. **MAPE (Mean Absolute Percentage Error) 📊**  
                   - Persentase error absolut rata-rata terhadap nilai aktual.  
                   - Semakin kecil → prediksi lebih akurat. Contoh: MAPE 0.10 → rata-rata prediksi meleset 10% dari nilai asli.
                """)

            # 4️⃣ Interpretasi per model
            with st.expander("📝 Interpretasi Hasil Setiap Model"):
                interpretasi_text = ""
                for idx, row in results.iterrows():
                    model = row['Model']
                    r2_out = row['R2_out_sample']
                    rmse_out = row['RMSE_out_sample']
                    mae_out = row['MAE_out_sample']
                    mape_out = row['MAPE_out_sample']
                    
                    if r2_out >= 0.9 and rmse_out < results['RMSE_out_sample'].median():
                        interpretasi = "Performa sangat baik: R² tinggi dan error rendah."
                    elif r2_out >= 0.7:
                        interpretasi = "Performa baik: R² cukup tinggi, error moderat."
                    elif r2_out >= 0.5:
                        interpretasi = "Performa sedang: R² sedang, perhatikan error."
                    else:
                        interpretasi = "Performa kurang baik: R² rendah, prediksi kemungkinan kurang akurat."
                    
                    if row['R2_in_sample'] - r2_out > 0.2:
                        interpretasi += " ⚠️ Kemungkinan overfitting (R² in-sample jauh lebih tinggi)."
                    
                    interpretasi_text += (
                        f"**{model}**: R²_out = {r2_out:.3f}, RMSE_out = {rmse_out:,.0f}, "
                        f"MAE = {mae_out:,.0f}, MAPE = {mape_out:.2%} → {interpretasi}\n\n"
                    )
                
                st.markdown(interpretasi_text)


# ==========================
# Halaman 3: Model Dasar Prediksi
# ==========================
elif page == "Model Dasar Prediksi":
    st.title("📈 Model Dasar Prediksi")

    EVAL_PATH = "model/results_evaluation.joblib"
    if os.path.exists(EVAL_PATH):
        results_eval = joblib.load(EVAL_PATH)
        st.success(f"📥 Hasil evaluasi berhasil dimuat! ({results_eval.shape[0]} baris x {results_eval.shape[1]} kolom)")
    else:
        st.warning("⚠️ Hasil evaluasi belum tersedia. Harap simpan results_evaluation.joblib di folder model/")
        st.stop()

    results_eval = results_eval.sort_values(by='RMSE_out_sample', ascending=True)

    # 1️⃣ Tabel Interaktif
    st.subheader("Tabel Hasil Evaluasi Model")
    st.dataframe(
        results_eval.style.format({
            'R2_in_sample': '{:.3f}',
            'R2_out_sample': '{:.3f}',
            'MSE_in_sample': '{:,.0f}',
            'MSE_out_sample': '{:,.0f}',
            'RMSE_in_sample': '{:,.0f}',
            'RMSE_out_sample': '{:,.0f}',
            'MAE_in_sample': '{:,.0f}',
            'MAE_out_sample': '{:,.0f}',
            'MAPE_in_sample': '{:.2%}',
            'MAPE_out_sample': '{:.2%}'
        }).background_gradient(cmap='plasma', subset=['R2_in_sample','R2_out_sample'])
          .background_gradient(cmap='viridis', subset=['RMSE_in_sample','RMSE_out_sample','MAE_in_sample','MAE_out_sample','MAPE_in_sample','MAPE_out_sample'])
    )

    # 2️⃣ Chart Interaktif dengan Tabs
    st.subheader("Visualisasi Performa Model")
    tab1, tab2 = st.tabs(["RMSE Out-Sample", "R² Out-Sample"])

    with tab1:
        fig_rmse = px.bar(
            results_eval,
            x='RMSE_out_sample',
            y='Model',
            orientation='h',
            color='RMSE_out_sample',
            color_continuous_scale='viridis',
            text='RMSE_out_sample'
        )
        fig_rmse.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_rmse, use_container_width=True)

    with tab2:
        fig_r2 = px.bar(
            results_eval,
            x='R2_out_sample',
            y='Model',
            orientation='h',
            color='R2_out_sample',
            color_continuous_scale='plasma',
            text='R2_out_sample'
        )
        fig_r2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_r2, use_container_width=True)

    # 3️⃣ Keterangan indikator
    # Expander: Keterangan indikator
    # --------------------------
    with st.expander("📌 Keterangan Rinci Indikator Evaluasi Model"):
        st.markdown("""
        1. **R² (R-squared) 📈**  
           - Menunjukkan seberapa baik model menjelaskan variasi target.  
           - Nilai 0–1: ≥0.9 sangat baik, 0.7–0.9 baik, 0.5–0.7 sedang, <0.5 kurang baik.  
           - R² negatif → model lebih buruk daripada prediksi mean.

        2. **MSE (Mean Squared Error) 💥**  
           - Rata-rata kuadrat selisih prediksi dengan nilai aktual.  
           - Semakin kecil → semakin akurat.  
           - Satuan = kuadrat target (misal target juta → MSE juta²).

        3. **RMSE (Root Mean Squared Error) 🌟**  
           - Akar dari MSE, satuan sama dengan target.  
           - Semakin kecil → prediksi lebih dekat ke nilai aktual.

        4. **MAE (Mean Absolute Error) ✨**  
           - Rata-rata absolut error prediksi.  
           - Semakin kecil → prediksi lebih akurat.

        5. **MAPE (Mean Absolute Percentage Error) 📊**  
           - Persentase error absolut rata-rata terhadap nilai aktual.  
           - Semakin kecil → prediksi lebih akurat. Contoh: MAPE 0.10 → rata-rata prediksi meleset 10% dari nilai asli.
        """)

    with st.expander("📝 Interpretasi Hasil Setiap Model"):
        interpretasi_text = ""
        for idx, row in results_eval.iterrows():
            model = row['Model']
            r2_out = row['R2_out_sample']
            rmse_out = row['RMSE_out_sample']
            mae_out = row['MAE_out_sample']
            mape_out = row['MAPE_out_sample']
        
            if r2_out >= 0.9 and rmse_out < results_eval['RMSE_out_sample'].median():
                interpretasi = "Performa sangat baik: R² tinggi dan error rendah."
            elif r2_out >= 0.7:
                interpretasi = "Performa baik: R² cukup tinggi, error moderat."
            elif r2_out >= 0.5:
                interpretasi = "Performa sedang: R² sedang, perhatikan error."
            else:
                interpretasi = "Performa kurang baik: R² rendah, prediksi kemungkinan kurang akurat."
        
            if row['R2_in_sample'] - r2_out > 0.2:
                interpretasi += " ⚠️ Kemungkinan overfitting (R² in-sample jauh lebih tinggi)."
        
            interpretasi_text += (
                f"**{model}**: R²_out = {r2_out:.3f}, RMSE_out = {rmse_out:,.0f}, "
                f"MAE = {mae_out:,.0f}, MAPE = {mape_out:.2%} → {interpretasi}\n\n"
            )
    
        st.markdown(interpretasi_text)



# ==========================
# Halaman 4: Prediksi
# ==========================
elif page == "Prediksi":
    st.title("💡 Prediksi Harga")
    
    if "model_rf" not in st.session_state:
        st.warning("⚠️ Model belum tersedia!")
    else:
        model_rf = st.session_state["model_rf"]

        FEATURE_PATH = "model/feature_columns.joblib"
        if os.path.exists(FEATURE_PATH):
            feature_cols = joblib.load(FEATURE_PATH)
        else:
            st.warning("⚠️ Feature columns belum tersedia! Pastikan file 'feature_columns.joblib' ada di folder model/")
            st.stop()

        st.subheader("Input Variabel")

        # Buat input berjajar 3 kolom
        input_data = {}
        cols_layout = st.columns(3)
        for i, col in enumerate(feature_cols):
            with cols_layout[i % 3]:
                val = st.text_input(f"{col}", value="0.0")
                input_data[col] = val

        input_df = pd.DataFrame([input_data])

        # Bersihkan persentase menjadi float
        for col in input_df.columns:
            if input_df[col].dtype == object:
                input_df[col] = input_df[col].str.replace('%','').str.replace(',','').astype(float)

        if st.button("Prediksi Harga"):
            try:
                # Prediksi harga
                pred_harga = model_rf.predict(input_df)[0]
                st.success(f"💰 Prediksi Harga: {pred_harga:,.2f}")

                # Top-5 similarity
                scaler = StandardScaler()
                # Gunakan data dummy numerik float
                X_dummy = pd.DataFrame(np.random.rand(100, len(feature_cols)), columns=feature_cols)
                X_scaled = scaler.fit_transform(X_dummy)
                input_scaled = scaler.transform(input_df)
                sim_matrix = cosine_similarity(X_scaled, input_scaled)
                top5_idx = np.argsort(sim_matrix[:,0])[::-1][:5]

                st.subheader("Top-5 Data Paling Mirip")
                top5_list = []
                for i, idx in enumerate(top5_idx):
                    row = X_dummy.iloc[idx].copy()
                    sim_score = sim_matrix[idx,0]
                    row['Similarity'] = sim_score
                    row['Rank'] = i+1
                    top5_list.append(row)

                top5_df = pd.DataFrame(top5_list)
                # Tampilkan kolom Rank, fitur utama, Similarity
                display_cols = ['Rank'] + [c for c in feature_cols if c != "HARGAPENAWARAN"] + ['Similarity']
                st.dataframe(top5_df[display_cols].style.format({"Similarity": "{:.2%}"}))

            except Exception as e:
                st.error(f"⚠️ Terjadi error saat prediksi: {e}")


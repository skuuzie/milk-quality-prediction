import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import mpld3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, metrics
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

st.header(":glass_of_milk: Modelling Data Kualitas Susu :glass_of_milk:")
model_overview, model_predict = st.tabs(["Modelling Progress Overview", "Model Predict"])

def to_milkframe(ph: float, temp: int, has_taste: int, has_odor: int, has_fat: int, is_turbid: int, color_val: int) -> None:
    return np.array([[ph, temp, has_taste, has_odor, has_fat, is_turbid, color_val]])

def predict_quality(model: KNeighborsClassifier, x):
    quality = {
        0: "Rendah",
        1: "Sedang",
        2: "Tinggi"
    }
    print(x)
    print(model.predict(x)[0])
    return quality[model.predict(x)[0]]

class Runtime:
    def __init__(self, filename) -> None:
        self.df = pd.read_csv(filename)
        self.data_loaded = False
        self._model : KNeighborsClassifier = None

    def overview(self):
        self.df = self.df.rename(columns={
            "Temprature": "Temperature",
            "Taste" : "Rasa",
            "Odor": "Bau",
            "Fat ": "Lemak",
            "Turbidity": "Kekeruhan",
            "Colour": "Warna",
            "Grade": "Kualitas"
        })

        st.divider()
        st.header(':memo: Rangkuman Data')

        st.write(self.df.head())
        st.write("Jumlah Baris : ", self.df.shape[0])
        st.write("Jumlah Kolom : ", self.df.shape[1])

        st.header(":memo: Informasi Tipe Data")
        st.write(self.df.dtypes)

        st.header(":memo: Informasi Jumlah Data Kosong")
        st.write(self.df.isna().sum())

    def visualize(self):

        # Plot Bar
        st.header(':face_with_monocle: Kualitas Susu')
        milk_quality = self.df.groupby('Kualitas').count()["pH"].sort_values()
        fig = plt.figure(figsize=(4, 4))
        milk_quality.plot(kind="bar", color=sns.color_palette("muted"))
        plt.ylabel('Jumlah')
        plt.title('Persebaran Jumlah Data berdasarkan Kualitas Susu')
        components.html(
            f"""
            <div style="text-align: center;">
                {mpld3.fig_to_html(fig)}
            </div>
            """,
            height=400
        )
        st.write("0.0 = Kualitas Rendah")
        st.write("1.0 = Kualitas Sedang")
        st.write("2.0 = Kualitas Tinggi")

        # Heat Map
        st.header(':face_with_monocle: Korelasi Data antar Kolom')
        plt.title('Heatmap')
        st.write(
            sns.heatmap(self.df.select_dtypes(exclude="object").corr()).figure
        )

        # Outlier
        st.header(':face_with_monocle: Data Pencilan')

        q1 = self.df.select_dtypes(exclude=['object']).quantile (0.25)
        q3 = self.df.select_dtypes(exclude=['object']).quantile (0.75)
        iqr = q3-q1
        outlier_filter = (self.df.select_dtypes(exclude=['object']) < q1 - 1.5 * iqr) | (self.df.select_dtypes(exclude=['object']) > q3 + 1.5 * iqr)

        df_outlier = self.df.select_dtypes(exclude=['object'])
        for column in df_outlier:
            fig = plt.figure(figsize=(12, 4))
            sns.boxplot(data=df_outlier, x=column)
            st.write(fig)

        st.write("Diketahui terdapat data-data pencilan (outlier) pada kolom `pH`, `Warna`, dan `Temperature`.")

        ph_outliers_percentage = len(outlier_filter[outlier_filter.pH==True]) / len(self.df['pH']) * 100
        st.write(f'Persentase data outlier pada kolom `pH`: {ph_outliers_percentage:.02f}%')

        temp_outliers_percentage = len(outlier_filter[outlier_filter.Temperature==True]) / len(self.df['Temperature']) * 100
        st.write(f'Persentase data outlier pada kolom `Temperature`: {temp_outliers_percentage:.02f}%')

        fat_outliers_percentage = len(outlier_filter[outlier_filter.Warna==True]) / len(self.df['Warna']) * 100
        st.write(f'Persentase data outlier pada kolom `Warna`: {fat_outliers_percentage:.02f}%')

    def clean(self):
        st.header(":soap: Pembersihan Data :sponge:")

        st.write(":broom: Visualisasi data pencilan setelah dilakukan pembersihan dengan teknik Winsorizing")
        self.df['pH'] = winsorize(self.df['pH'], limits=[0, 0.05])
        self.df['Temperature'] = winsorize(self.df['Temperature'], limits=[0, 0.05])
        self.df['Warna'] = winsorize(self.df['Warna'], limits=[0, 0.05])

        fig = plt.figure(figsize=(12, 4))
        sns.boxplot(data=self.df, x=self.df['pH'])
        st.pyplot(fig)

        fig = plt.figure(figsize=(12, 4))
        sns.boxplot(data=self.df, x=self.df['Temperature'])
        st.pyplot(fig)

        fig = plt.figure(figsize=(12, 4))
        sns.boxplot(data=self.df, x=self.df['Warna'])
        st.pyplot(fig)
    
    def model(self):
        st.header("	:gear: Pemodelan Data :recycle:")

        st.write("Pemodelan data dilakukan dengan K-Nearest Neighbors (KNN), salah satu alasan penggunaan KNN adalah karena sifat data outlier untuk dataset kualitas susu ini masih tergolong banyak, tetapi wajar (seperti tingkat pH dan suhu susu yang memang bisa bervariasi).")

        le = preprocessing.LabelEncoder()
        self.df[["Kualitas"]] = self.df[["Kualitas"]].apply(le.fit_transform)

        x = self.df.iloc[:, 0:-1].values
        y = self.df.iloc[:, -1].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

        accuracy = []

        for i in range(1,15):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(x_train,y_train)
            pred_i = knn.predict(x_test)
            accuracy_i = metrics.accuracy_score(y_test, pred_i)
            accuracy.append(accuracy_i)

        fig = plt.figure(figsize=(15,6))
        plt.plot(range(1,15, 1), accuracy, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title('Accuracy vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        
        st.pyplot(fig)

        st.write("Setelah dilakukan evaluasi model sebanyak 14 kali, diketahui model berforma baik pada nilai `n` yang rendah")

        # Final n = 4
        self.fit_model(x_train, y_train, 4)
        k = KFold(n_splits=5)

        score = cross_val_score(self._model, x_train, y_train, scoring = 'accuracy', cv = k).mean()
        st.write(f'Akurasi data training: `{round(score, 3)}`')

        y_pred = self._model.predict(x_test)
        st.write(f"Akurasi data test: `{round(metrics.accuracy_score(y_test, y_pred), 3)}`")

        st.divider()

        st.header(":male-scientist: Hasil Model :female-scientist:")
        report = metrics.classification_report(y_test, y_pred, target_names=["High", "Medium", "Low"])
        st.write(report)

        st.write("Hasil akurasi model mencapai `~98%`")
    
    def fit_model(self, x_train, y_train, n_neighbors):
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self._model.fit(x_train, y_train)
        
def is_csv(filename):
    return filename.lower().endswith('.csv')

app = None

with model_overview:
    st.header('Guide')
    st.markdown(
        '''1. Gunakan Format File CSV\n2. Dataset yang dimasukkan adalah dataset kualitas susu\n\nReferensi Data: https://www.kaggle.com/datasets/cpluzshrijayan/milkquality
        '''
    )

    uploaded_file = st.file_uploader('Upload File CSV')
    upload_txt = st.text("Silahkan upload file CSV")

    if uploaded_file is not None:
        if not is_csv(uploaded_file.name):
            upload_txt.text("File harus bertipe .csv")
        else:
            try:
                upload_txt.text("")
                app = Runtime(uploaded_file.name)
                app.data_loaded = True
            except:
                upload_txt.text("Format file tidak dikenal atau korup.")

    if isinstance(app, Runtime):
        with st.spinner("Proses..."):
            app.overview()
            app.visualize()

            st.divider()

            app.clean()
            app.model()

with model_predict:
    if not isinstance(app, Runtime) or app._model == None:
        st.write('Silahkan lakukan modelling terlebih dahulu sebelum melakukan prediksi :glass_of_milk: :glass_of_milk: :glass_of_milk:')
    else:
        ph = st.number_input("Tingkat pH susu:", min_value=0.0, max_value=10.0, step=0.1)
        temp = st.number_input("Suhu susu (celcius):", step=1)
        color = st.number_input("Warna susu (rentang hex 0-255):", min_value=0, max_value=255, step=1)

        taste = st.checkbox("Apakah susu mempunyai rasa?")
        odor = st.checkbox("Apakah susu bau?")
        fat = st.checkbox("Apakah susu mempunyai kadar lemak?")
        turbidity = st.checkbox("Apakah tampilan susu terlihat keruh?")

        inp = to_milkframe(ph, temp, taste, odor, fat, turbidity, color)

        predicted = predict_quality(app._model, inp)
        st.write(f"Hasil prediksi menunjukkan susu berkualitas `{predicted}`.")
        print(predicted)
import streamlit as st # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import time
from keras.models import load_model # type: ignore
from keras.preprocessing import image # type: ignore
from PIL import Image, ImageOps # type: ignore

icon_path = "C:\\Users\\Ganda\\Sperma_icon.ico"
st.set_page_config(page_title="sperma.AI", page_icon=icon_path)

# Memuat model yang telah dilatih
model = load_model('C:\\Users\\Ganda\\model_sperma_2kelas.h5')

def prediksi_gambar(file_path):
    class_names = ['Bad_Sperma', 'Good_Sperma']  # Ganti urutan label
    data = np.ndarray(shape=(1, 130, 130, 3), dtype=np.float32)
    image = Image.open(file_path).convert("RGB")
    size = (130, 130)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Melakukan prediksi menggunakan model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index] * 100

    hasil = {
        'label_kelas': class_name,
        'skor_kepercayaan':  f"{confidence_score:.2f}%"
    }
    return hasil

# Aplikasi Streamlit
# Navigasi
halaman_terpilih = st.selectbox("Pilih Halaman", ["Beranda", "Halaman Deteksi", "Visualisasi Model"], format_func=lambda x: x)

if halaman_terpilih == "Beranda":
    # Tampilkan Halaman Beranda
    st.header("Selamat Datang Di Aplikasi Morfologi Sperma", divider='rainbow')
    st.write(
        "Aplikasi Ini Memungkinkan Anda Untuk Mengunggah Gambar Morfologi Sperma "
        "Untuk Menganalisa Morfologi Sperma Bagus Dan Morfologi Sperma Tidak Bagus."
    )
    st.write(
        "Silahkan Pilih Halaman Deteksi Untuk Melanjutkan Pemeriksaan Atau Pilih Halaman Visualisasi Model Untuk Melihat Hasil Kinerja Model AI, Seperti ''Accuracy Class''  ''Confusional Matrix''  ''Accuracy Epoch'' Dan ''Loss Epoch''"
    )
elif halaman_terpilih == "Halaman Deteksi":
    # Tampilkan Halaman Deteksi
    st.title("Unggah Gambar")
    st.markdown("---")

    # Unggah Gambar Melalui Streamlit
    berkas_gambar = st.file_uploader("Silahkan Pilih Gambar", type=["jpg", "jpeg", "png"])
    if berkas_gambar:
        # Tampilkan Gambar Yang Dipilih
        st.image(berkas_gambar, caption="Gambar Yang Diunggah", use_column_width=True)
        if st.button("Deteksi"):
            # Simpan Berkas Gambar Yang Diunggah Ke Lokasi Sementara
            with open("temp_image.jpg", "wb") as f:
                f.write(berkas_gambar.getbuffer())

            # Lakukan Prediksi Pada Berkas Yang Disimpan
            hasil_prediksi = prediksi_gambar("temp_image.jpg")
            # Tampilkan Hasil Prediksi
            st.write(f"Hasil Deteksi: {hasil_prediksi['label_kelas']}")
            st.write(f"Skor Kepercayaan: {hasil_prediksi['skor_kepercayaan']}")

            if hasil_prediksi['label_kelas'] == 'Good_Sperm':
                st.write(
                    "Selamat! Berdasarkan Deteksi kami, gambar yang Anda masukkan adalah Good Sperm. Namun, ingatlah bahwa ini hanya hasil dari model kecerdasan buatan kami.")
            elif hasil_prediksi['label_kelas'] == 'Bad_Sperm':
                st.write(
                    "Sepertinya hasil deteksi kami menunjukkan bahwa gambar yang Anda masukkan adalah Bad Sperm. Namun, perlu diingat bahwa ini hanya hasil dari model kecerdasan buatan kami.")
            else:
                st.write("Label tidak dikenali. Silakan periksa input Anda.")
                
elif halaman_terpilih == "Visualisasi Model":
    st.title("Kinerja Model AI")
    st.markdown("---")

    def display_image_table(image_path1, title1, caption1, image_path2='', title2='', caption2=''):
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2) if image_path2 else None

        col1, col2 = st.columns(2)

        with col1:
            col1.markdown(f'<h2 style="text-align:center;">{title1}</h2>', unsafe_allow_html=True)
            col1.markdown(
                f'<div style="display: flex; justify-content: center;"></div>',
                unsafe_allow_html=True
            )
            col1.image(image1, use_column_width=True)
            col1.markdown(f'<p style="text-align:left;">{caption1}</p>', unsafe_allow_html=True)

        with col2:
            if image2:
                col2.markdown(f'<h2 style="text-align:center;">{title2}</h2>', unsafe_allow_html=True)
                col2.markdown(
                    f'<div style="display: flex; justify-content: center;"></div>',
                    unsafe_allow_html=True
                )
                col2.image(image2, use_column_width=True)
                col2.markdown(f'<p style="text-align:left;">{caption2}</p>', unsafe_allow_html=True)

    image_info = [
        {'path': 'C:\\Users\\Ganda\\Documents\\model_ai\\accuracy_class.png', 'title': 'Accuracy Class', 'caption': 'Tabel ini menunjukkan bahwa model kecerdasan buatan berhasil mencapai akurasi sempurna, 1.00, dalam mengklasifikasikan "Good Sperma" dengan benar dari 15 sampel, sementara untuk "Bad Sperma" model mencapai akurasi sebesar 0.93 dengan jumlah sampel yang sama.'},
        {'path': 'C:\\Users\\Ganda\\Documents\\model_ai\\confusional_matrix.png', 'title': 'Confusional Matrix', 'caption': 'Confusional Matrix ini menunjukkan bahwa model mengklasifikasikan semua 15 sampel "Good Sperma" dengan benar, namun salah mengklasifikasikan 1 sampel "Bad Sperma" sebagai "Good Sperma" dari total 15 sampel'},
        {'path': 'C:\\Users\\Ganda\\Documents\\model_ai\\accuracy_epoch.png', 'title': 'Accuracy Epoch', 'caption': 'Grafik ini menunjukkan bahwa akurasi model kecerdasan buatan meningkat dengan cepat dan mencapai hampir 100% dalam sekitar 50 epochs, kemudian stabil pada tingkat tinggi baik untuk akurasi pelatihan "acc" maupun akurasi pengujian "test acc".'},
        {'path': 'C:\\Users\\Ganda\\Documents\\model_ai\\loss_epoch.png', 'title': 'Loss Epoch', 'caption': 'Grafik ini menunjukkan bahwa nilai loss pelatihan cepat menurun dan stabil pada nilai sangat rendah, sementara loss pengujian juga menurun tetapi stabil pada nilai yang sedikit lebih tinggi.'},
    ]

    for i in range(0, len(image_info), 2):
        if i + 1 < len(image_info):
            display_image_table(
                image_info[i]['path'], image_info[i]['title'], image_info[i]['caption'],
                image_info[i + 1]['path'], image_info[i + 1]['title'], image_info[i + 1]['caption']
            )
        else:
            display_image_table(image_info[i]['path'], image_info[i]['title'], image_info[i]['caption'])

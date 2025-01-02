import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Fungsi untuk memuat model dari file
@st.cache
def load_model():
    # Memuat model yang sudah disimpan
    model = tf.keras.models.load_model('model.h5')  # Pastikan model disimpan dengan nama yang benar
    return model

# Fungsi untuk melakukan prediksi pada gambar
def predict_image(model, image):
    # Resize image sesuai dengan ukuran input model
    image = image.resize((128, 128))
    img_array = np.array(image)  # Convert image to array
    img_array = img_array.astype('float32') / 255.0  # Normalize image to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediksi menggunakan model
    predictions = model.predict(img_array)
    return predictions

# Fungsi utama aplikasi Streamlit
def main():
    st.title('Prediksi Penyakit Tanaman')
    st.write('Upload gambar daun untuk mengetahui kondisi tanaman.')

    # Mengupload gambar
    uploaded_file = st.file_uploader("Pilih gambar daun", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Menampilkan gambar yang di-upload
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang di-upload', use_column_width=True)

        # Memuat model
        model = load_model()

        # Prediksi gambar
        if st.button('Prediksi'):
            predictions = predict_image(model, image)

            class_names = ['Cordana', 'Fusarium', 'Healthy Leaf', 'Pestalotiopsis', 'Sigatoka']
            result = {}

            for i, class_name in enumerate(class_names):
                result[class_name] = 'Terinfeksi' if predictions[0][i] > 0.5 else 'Sehat'

            # Menampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            for class_name, condition in result.items():
                st.write(f"{class_name}: {condition}")

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    main()

import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import gdown
import zipfile

st.set_page_config(page_title="Görsel Öneri Sistemi", layout="wide")
st.title("Görsel Tabanlı Benzer Ürün Öneri Sistemi")

# Google Drive Dosya ID'leri

FILENAMES_FILE_ID = "172UU0JLRZAucn94hCqg87IaJm48MF8Mk"
FEATURES_FILE_ID = "1ERFHzUyt7jepHNQH9Dvpt5KRJJ8X0GUM"
IMAGES_ZIP_FILE_ID = "1D06CuDiiL7Yd4C4WVKVShZQ1fog-oX1z" 

@st.cache_resource
def load_data_and_images():
    # 1. filenames.pkl indir
    if not os.path.exists('filenames.pkl'):
        url = f'https://drive.google.com/uc?id={FILENAMES_FILE_ID}'
        gdown.download(url, 'filenames.pkl', quiet=False)
        
    # 2. Images_features.pkl indir
    if not os.path.exists('Images_features.pkl'):
        url = f'https://drive.google.com/uc?id={FEATURES_FILE_ID}'
        gdown.download(url, 'Images_features.pkl', quiet=False)

    # 3. images klasörü yoksa zip olarak indir ve aç (Sadece ilk kurulumda çalışır)
    if not os.path.exists('images'):
        url = f'https://drive.google.com/uc?id={IMAGES_ZIP_FILE_ID}'
        gdown.download(url, 'images.zip', quiet=False)
        with zipfile.ZipFile('images.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove('images.zip')

    filenames = pickle.load(open('filenames.pkl', 'rb'))
    features = pickle.load(open('Images_features.pkl', 'rb'))
    return filenames, features

with st.spinner('Model verileri ve görseller yükleniyor, lütfen bekleyin...'):
    filenames, features = load_data_and_images()

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(features)

selected_image = st.selectbox("Lütfen bir ürün görseli seçin:", filenames)

if selected_image:
    fixed_path = selected_image.replace('\\', '/')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Seçilen Ürün")
        if os.path.exists(fixed_path):
            st.image(fixed_path, use_container_width=True)
        else:
            st.warning(f"Görsel sunucuda bulunamadı: {fixed_path}")

    selected_index = filenames.index(selected_image)
    distances, indices = neighbors.kneighbors([features[selected_index]])

    with col2:
        st.subheader("Benzer Önerilen Ürünler")
        cols = st.columns(5)
        for i, idx in enumerate(indices[0][1:]):
            recommended_file = filenames[idx]
            fixed_rec_path = recommended_file.replace('\\', '/')
            with cols[i]:
                if os.path.exists(fixed_rec_path):
                    st.image(fixed_rec_path, use_container_width=True)
                    st.caption(f"Mesafe: {distances[0][i+1]:.2f}")
                else:
                    st.write("Görsel yok")

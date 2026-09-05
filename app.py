import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Görsel Öneri Sistemi", layout="wide")
st.title("Görsel Tabanlı Benzer Ürün Öneri Sistemi")

@st.cache_resource
def load_data():
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    features = pickle.load(open('Images_features.pkl', 'rb'))
    return filenames, features

filenames, features = load_data()

# NearestNeighbors modelini tanımla ve eğit
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(features)

# Kullanıcıya listeden seçim yaptırma
selected_image = st.selectbox("Lütfen bir ürün görseli seçin:", filenames)

if selected_image:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Seçilen Ürün")
        if os.path.exists(selected_image):
            st.image(selected_image, use_container_width=True)
        else:
            st.warning(f"Görsel bulunamadı: {selected_image}")

    # Seçilen görselin indeksini bul ve benzerleri hesapla
    selected_index = filenames.index(selected_image)
    distances, indices = neighbors.kneighbors([features[selected_index]])

    with col2:
        st.subheader("Benzer Önerilen Ürünler")
        # 5 adet benzer ürünü yan yana göster (ilk eleman kendisidir, o yüzden [1:] ile başlanır)
        cols = st.columns(5)
        for i, idx in enumerate(indices[0][1:]):
            recommended_file = filenames[idx]
            with cols[i]:
                if os.path.exists(recommended_file):
                    st.image(recommended_file, use_container_width=True)
                    st.caption(f"Mesafe: {distances[0][i+1]:.2f}")
                else:
                    st.write("Görsel yok")

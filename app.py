import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import gdown
import tensorflow as tf # ResNet50 için gerekli (özellik çıkarmak istiyorsanız)

st.set_page_config(page_title="Görsel Öneri Sistemi", layout="wide")
st.title("Görsel Tabanlı Benzer Ürün Öneri Sistemi")

# Google Drive Dosya ID'leri 
FILENAMES_FILE_ID = "172UU0JLRZAucn94hCqg87IaJm48MF8Mk"
FEATURES_FILE_ID = "1ERFHzUyt7jepHNQH9Dvpt5KRJJ8X0GUM"

@st.cache_resource
def load_data():
    if not os.path.exists('filenames.pkl'):
        url = f'https://drive.google.com/uc?id={FILENAMES_FILE_ID}'
        gdown.download(url, 'filenames.pkl', quiet=False)
        
    if not os.path.exists('Images_features.pkl'):
        url = f'https://drive.google.com/uc?id={FEATURES_FILE_ID}'
        gdown.download(url, 'Images_features.pkl', quiet=False)

    filenames = pickle.load(open('filenames.pkl', 'rb'))
    features = pickle.load(open('Images_features.pkl', 'rb'))
    return filenames, features

with st.spinner('Model verileri yükleniyor, lütfen bekleyin...'):
    filenames, features = load_data()

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(features)

# --- SEÇİM YÖNTEMİ: Dosya Yükletme ---
st.subheader("Ürün Bul")
uploaded_file = st.file_uploader("Lütfen aratmak istediğiniz ürün görselini yükleyin:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Yüklenen görseli ekranda göster
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Yüklenen Görsel")
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, use_container_width=True)

    # 2. Yüklenen görselin özelliklerini çıkarma (Modeliniz ve ön işleme adımlarınız burada olmalı)
    # NOT: Eğer yüklenen görselin özelliklerini anlık çıkaracaksanız model.predict() kullanmalısınız.
    # Örnek mantık:
    # processed_img = preprocess(uploaded_image)
    # query_feature = model.predict(processed_img)
    
    # DİKKAT: Eğer sisteminiz "dataset içindeki bir resmi seçme" mantığıyla çalışıyorsa, 
    # yüklenen resmi doğrudan features matrisinde aratamayız. Bunun yerine model ile feature üretmelisiniz.
    
    # Alternatif olarak eğer hala selectbox kullanmak istiyorsanız ve sadece "görselleri sunucuda aramak" yerine 
    # projenin yerelde çalışmasını istiyorsanız, images klasörünü proje dizinine (kodun yanına) atabilirsiniz.

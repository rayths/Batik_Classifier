from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F

logo = Image.open('assets/Logo.png')
st.set_page_config(
    page_title='KNOB 😱', 
    page_icon=logo, 
    layout='wide'
)

selected_tab = option_menu(
    menu_title=None,
    options=["Upload", "Take a Photo"],
    icons=["upload", "camera"],
    menu_icon="cast",
    default_index=0,
    key="nav", 
    orientation="horizontal"
)

# Load pre-trained PyTorch model
@st.cache_data()
def load_model():
    try:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 20)
        model_path = r'batik_classifier_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model tidak ditemukan di path: {model_path}")
        return None

model = load_model()

if model is None:
    st.stop()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Image transformations
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Mapping untuk nama kelas Batik
class_names = ['Batik Bali', 'Batik Betawi', 'Batik Celup', 'Batik Cendrawasih', 
               'Batik Ceplok', 'Batik Ciamis', 'Batik Garutan', 'Batik Gentongan', 
               'Batik Kawung', 'Batik Keraton', 'Batik Lasem', 'Batik Mega Mendung', 
               'Batik Parang', 'Batik Pekalongan', 'Batik Priangan', 'Batik Sekar', 
               'Batik Sidoluhur', 'Batik Sidomukti', 'Batik Sogan', 'Batik Tambal']

# Create int_label dictionary
int_label = {i: class_name for i, class_name in enumerate(class_names)}

def predict_uploaded_image(uploaded_file, model, transform, int_label):
    """Predicts the class of an uploaded image using the provided model."""
    image = Image.open(uploaded_file).convert('RGB') 
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        output = model(image)
        prediction = torch.argmax(F.softmax(output, dim=1)).cpu().item()
        predicted_class = int_label[prediction]
    return predicted_class

# Informasi tambahan tentang kelas batik 
batik_info = {
    'Batik Bali': "Batik Bali umumnya memiliki motif yang terinspirasi dari alam dan mitologi. Batik ini berasal dari daerah Bali.",
    'Batik Lasem': "Batik Lasem dikenal dengan warna-warna cerah dan motif yang dipengaruhi budaya China. Batik ini berasal dari daerah Rembang, Jawa Tengah.",
    'Batik Betawi': "Batik Betawi memiliki motif yang beragam, seringkali menggambarkan flora dan fauna. Batik ini berasal dari daerah Jakarta.",
    'Batik Mega Mendung': "Batik Mega Mendung khas dengan motif awan bergaya Tiongkok. Batik ini berasal dari daerah Cirebon, Jawa Barat.",
    'Batik Celup': "Batik Celup dibuat dengan teknik celup ikat, menghasilkan motif abstrak dan dinamis. Batik ini banyak digunakan diberbagai daerah di Indonesia.",
    'Batik Parang': "Batik Parang memiliki motif huruf 'S' yang saling terkait, melambangkan kontinuitas. Batik ini berasal dari daerah Yogyakarta dan Solo.",
    'Batik Cendrawasih': "Batik Cendrawasih menampilkan motif burung cendrawasih yang indah dan eksotis. Batik ini berasal dari daerah Papua.",
    'Batik Pekalongan': "Batik Pekalongan dikenal dengan warna-warna cerah dan motif yang rumit. Batik ini berasal dari daerah Pekalongan, Jawa Tengah.",
    'Batik Ceplok': "Batik Ceplok memiliki motif geometris, seperti lingkaran, kotak, dan bintang. Batik ini berasal dari daerah Bantul, Yogyakarta.",
    'Batik Priangan': "Batik Priangan memiliki warna-warna lembut dan motif yang naturalis. Batik ini berasal dari daerah Jawa Barat dan Banten.",
    'Batik Ciamis': "Batik Ciamis dikenal dengan motif ayam jago dan tumbuhan paku. Batik ini berasal dari daerah Ciamis, Jawa Barat.",
    'Batik Sekar': "Batik Sekar memiliki motif bunga yang beragam dan indah. Batik ini berasal dari daerah Solo dan Yoyakarta.",
    'Batik Garutan': "Batik Garutan terkenal dengan motif lereng dan kombinasi warna yang khas. Batik ini berasal dari daerah Garut, Jawa Barat.",
    'Batik Sidoluhur': "Batik Sidoluhur memiliki motif yang rumit dan filosofis, sering digunakan dalam upacara adat. Batik ini berasal dari daerah Yoyakarta dan Surakarta.",
    'Batik Gentongan': "Batik Gentongan diproses dengan direndam dalam gentong, menghasilkan warna yang khas. Batik ini berasal dari daerah Madura.",
    'Batik Sidomukti': "Batik Sidomukti memiliki motif yang melambangkan kemakmuran dan kesejahteraan. Batik ini berasal dari daerah Solo dan Yogyakarta.",
    'Batik Kawung': "Batik Kawung memiliki motif bulatan yang melambangkan keadilan dan keseimbangan. Batik ini berasal dari daerah Yogyakarta.",
    'Batik Sogan': "Batik Sogan memiliki warna cokelat yang khas dan motif yang sederhana. Batik ini berasal dari daerah Solo dan Yogyakarta.",
    'Batik Keraton': "Batik Keraton memiliki motif yang sakral dan hanya boleh dikenakan oleh keluarga kerajaan. Batik ini berasal dari daerah Yogyakarta dan Surakarta.",
    'Batik Tambal': "Batik Tambal dibuat dengan menggabungkan potongan-potongan kain perca. Batik ini berasal dari daerah Yogyakarta."
}

if selected_tab == "Upload": 
    # Membuat form untuk upload file
    uploaded_file = st.file_uploader("Pilih gambar batik Anda", type=["jpg", "png", "jpeg"])

    # Memprediksi jenis batik
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

        if st.button('Prediksi'):
            with st.spinner('Sedang memprediksi...'):
                predicted_class = predict_uploaded_image(uploaded_file, model, val_transform, int_label) 

            st.write(f'Prediksi: **{predicted_class}**')

            # Menampilkan informasi tambahan
            if predicted_class in batik_info:
                st.info(batik_info[predicted_class])

elif selected_tab == "Take a Photo":
    # Membuat form untuk mengambil foto
    picture = st.camera_input("Ambil foto batik Anda")

    # Memprediksi jenis batik
    if picture is not None:
        image = Image.open(picture).convert('RGB')
        st.image(image, caption='Gambar yang diambil', use_column_width=True)

        if st.button('Prediksi Foto'):
            with st.spinner('Sedang memprediksi...'):
                predicted_class = predict_uploaded_image(picture, model, val_transform, int_label)

            st.write(f'Prediksi: **{predicted_class}**')

            # Menampilkan informasi tambahan
            if predicted_class in batik_info:
                st.info(batik_info[predicted_class])
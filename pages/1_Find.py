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
    'Batik Bali': "Batik Bali umumnya memiliki motif yang terinspirasi dari alam dan mitologi.",
    'Batik Lasem': "Batik Lasem dikenal dengan warna-warna cerah dan motif yang dipengaruhi budaya China.",
    'Batik Betawi': "Batik Betawi memiliki motif yang beragam, seringkali menggambarkan flora dan fauna.",
    'Batik Mega Mendung': "Batik Mega Mendung khas Cirebon dengan motif awan bergaya Tiongkok.",
    'Batik Celup': "Batik Celup dibuat dengan teknik celup ikat, menghasilkan motif abstrak dan dinamis.",
    'Batik Parang': "Batik Parang memiliki motif huruf 'S' yang saling terkait, melambangkan kontinuitas.",
    'Batik Cendrawasih': "Batik Cendrawasih menampilkan motif burung cendrawasih yang indah dan eksotis.",
    'Batik Pekalongan': "Batik Pekalongan dikenal dengan warna-warna cerah dan motif yang rumit.",
    'Batik Ceplok': "Batik Ceplok memiliki motif geometris, seperti lingkaran, kotak, dan bintang.",
    'Batik Priangan': "Batik Priangan memiliki warna-warna lembut dan motif yang naturalis.",
    'Batik Ciamis': "Batik Ciamis dikenal dengan motif ayam jago dan tumbuhan paku.",
    'Batik Sekar': "Batik Sekar memiliki motif bunga yang beragam dan indah.",
    'Batik Garutan': "Batik Garutan terkenal dengan motif lereng dan kombinasi warna yang khas.",
    'Batik Sidoluhur': "Batik Sidoluhur memiliki motif yang rumit dan filosofis, sering digunakan dalam upacara adat.",
    'Batik Gentongan': "Batik Gentongan diproses dengan direndam dalam gentong, menghasilkan warna yang khas.",
    'Batik Sidomukti': "Batik Sidomukti memiliki motif yang melambangkan kemakmuran dan kesejahteraan.",
    'Batik Kawung': "Batik Kawung memiliki motif bulatan yang melambangkan keadilan dan keseimbangan.",
    'Batik Sogan': "Batik Sogan memiliki warna cokelat yang khas dan motif yang sederhana.",
    'Batik Keraton': "Batik Keraton memiliki motif yang sakral dan hanya boleh dikenakan oleh keluarga kerajaan.",
    'Batik Tambal': "Batik Tambal dibuat dengan menggabungkan potongan-potongan kain perca."
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
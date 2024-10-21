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
    'Batik Bali': {
        'deskripsi': "Batik Bali umumnya memiliki motif yang terinspirasi dari alam dan mitologi. Batik ini berasal dari daerah Bali.",
        'makna': "Menggambarkan keseimbangan antara manusia dan alam, terinspirasi dari flora dan fauna Bali serta mitologi Hindu.",
        'asal': "Bali",
        'rekomendasi_acara': "Cocok digunakan pada acara keagamaan, pernikahan, atau upacara adat di Bali."
    },
    'Batik Lasem': {
        'deskripsi': "Batik Lasem dikenal dengan warna-warna cerah dan motif yang dipengaruhi budaya China. Batik ini berasal dari daerah Rembang, Jawa Tengah.",
        'makna': "Memadukan budaya Tionghoa dan Jawa dengan simbol-simbol keberuntungan dan kesejahteraan.",
        'asal': "Lasem, Rembang, Jawa Tengah",
        'rekomendasi_acara': "Sesuai untuk acara budaya, perayaan Imlek, atau acara formal dengan nuansa budaya Tionghoa."
    },
    'Batik Betawi': {
        'deskripsi': "Batik Betawi memiliki motif yang beragam, seringkali menggambarkan flora dan fauna. Batik ini berasal dari daerah Jakarta.",
        'makna': "Mencerminkan kebhinekaan budaya Betawi, dengan motif flora dan fauna khas Jakarta.",
        'asal': "Jakarta",
        'rekomendasi_acara': "Cocok untuk acara-acara tradisional Betawi seperti pernikahan adat Betawi dan festival budaya."
    },
    'Batik Mega Mendung': {
        'deskripsi': "Batik Mega Mendung khas dengan motif awan bergaya Tiongkok. Batik ini berasal dari daerah Cirebon, Jawa Barat.",
        'makna': "Melambangkan ketenangan dan keseimbangan hidup melalui motif awan.",
        'asal': "Cirebon, Jawa Barat",
        'rekomendasi_acara': "Sesuai untuk acara keagamaan, seminar, atau pertemuan formal."
    },
    'Batik Parang': {
        'deskripsi': "Batik Parang memiliki motif huruf 'S' yang saling terkait, melambangkan kontinuitas. Batik ini berasal dari daerah Yogyakarta dan Solo.",
        'makna': "Mengandung filosofi kekuatan dan ketangguhan, dengan motif 'S' yang saling berkelindan.",
        'asal': "Yogyakarta dan Solo",
        'rekomendasi_acara': "Cocok digunakan pada acara adat, pernikahan tradisional, atau sebagai simbol kehormatan."
    },
    'Batik Cendrawasih': {
        'deskripsi': "Batik Cendrawasih menampilkan motif burung cendrawasih yang indah dan eksotis. Batik ini berasal dari daerah Papua.",
        'makna': "Melambangkan keindahan dan keanggunan melalui gambar burung Cendrawasih.",
        'asal': "Papua",
        'rekomendasi_acara': "Cocok untuk acara budaya Papua, pameran seni, atau perayaan nasional."
    },
    'Batik Pekalongan': {
        'deskripsi': "Batik Pekalongan dikenal dengan warna-warna cerah dan motif yang rumit. Batik ini berasal dari daerah Pekalongan, Jawa Tengah.",
        'makna': "Melambangkan kreativitas dan inovasi dengan motif yang rumit dan warna-warna cerah.",
        'asal': "Pekalongan, Jawa Tengah",
        'rekomendasi_acara': "Sesuai untuk acara formal, pameran batik, atau acara kebudayaan."
    },
    'Batik Ceplok': {
        'deskripsi': "Batik Ceplok memiliki motif geometris, seperti lingkaran, kotak, dan bintang. Batik ini berasal dari daerah Bantul, Yogyakarta.",
        'makna': "Melambangkan keteraturan dan keseimbangan dengan pola geometris yang harmonis.",
        'asal': "Bantul, Yogyakarta",
        'rekomendasi_acara': "Cocok untuk acara formal, pameran seni, atau upacara adat."
    },
    'Batik Priangan': {
        'deskripsi': "Batik Priangan memiliki warna-warna lembut dan motif yang naturalis. Batik ini berasal dari daerah Jawa Barat dan Banten.",
        'makna': "Melambangkan keindahan alam dan ketenangan hidup.",
        'asal': "Jawa Barat dan Banten",
        'rekomendasi_acara': "Sesuai untuk acara pernikahan, perayaan budaya, atau acara formal lainnya."
    },
    'Batik Ciamis': {
        'deskripsi': "Batik Ciamis dikenal dengan motif ayam jago dan tumbuhan paku. Batik ini berasal dari daerah Ciamis, Jawa Barat.",
        'makna': "Menggambarkan kekuatan dan keindahan alam.",
        'asal': "Ciamis, Jawa Barat",
        'rekomendasi_acara': "Cocok digunakan pada acara budaya lokal atau pernikahan adat."
    },
    'Batik Sekar': {
        'deskripsi': "Batik Sekar memiliki motif bunga yang beragam dan indah. Batik ini berasal dari daerah Solo dan Yogyakarta.",
        'makna': "Melambangkan keindahan, kesuburan, dan kehidupan yang berlimpah.",
        'asal': "Solo dan Yogyakarta",
        'rekomendasi_acara': "Cocok untuk acara pernikahan, festival budaya, atau acara formal lainnya."
    },
    'Batik Garutan': {
        'deskripsi': "Batik Garutan terkenal dengan motif lereng dan kombinasi warna yang khas. Batik ini berasal dari daerah Garut, Jawa Barat.",
        'makna': "Melambangkan keuletan dan ketekunan dalam kehidupan.",
        'asal': "Garut, Jawa Barat",
        'rekomendasi_acara': "Sesuai untuk acara formal, pameran batik, atau perayaan budaya."
    },
    'Batik Sidoluhur': {
        'deskripsi': "Batik Sidoluhur memiliki motif yang rumit dan filosofis, sering digunakan dalam upacara adat. Batik ini berasal dari daerah Yogyakarta dan Surakarta.",
        'makna': "Melambangkan harapan akan kedudukan yang luhur dan mulia dalam kehidupan.",
        'asal': "Yogyakarta dan Surakarta",
        'rekomendasi_acara': "Cocok untuk upacara adat dan pernikahan tradisional."
    },
    'Batik Gentongan': {
        'deskripsi': "Batik Gentongan diproses dengan direndam dalam gentong, menghasilkan warna yang khas. Batik ini berasal dari daerah Madura.",
        'makna': "Mencerminkan ketekunan dan kerajinan dalam pembuatan batik.",
        'asal': "Madura",
        'rekomendasi_acara': "Sesuai untuk acara formal, pameran seni, atau festival budaya."
    },
    'Batik Sidomukti': {
        'deskripsi': "Batik Sidomukti memiliki motif yang melambangkan kemakmuran dan kesejahteraan. Batik ini berasal dari daerah Solo dan Yogyakarta.",
        'makna': "Melambangkan harapan untuk hidup yang makmur dan sejahtera.",
        'asal': "Solo dan Yogyakarta",
        'rekomendasi_acara': "Cocok digunakan pada acara pernikahan adat dan upacara resmi."
    },
    'Batik Kawung': {
        'deskripsi': "Batik Kawung memiliki motif bulatan yang melambangkan keadilan dan keseimbangan. Batik ini berasal dari daerah Yogyakarta.",
        'makna': "Melambangkan keadilan, keseimbangan, dan kebijaksanaan.",
        'asal': "Yogyakarta",
        'rekomendasi_acara': "Cocok untuk acara resmi, upacara adat, atau pertemuan formal."
    },
    'Batik Sogan': {
        'deskripsi': "Batik Sogan memiliki warna cokelat yang khas dan motif yang sederhana. Batik ini berasal dari daerah Solo dan Yogyakarta.",
        'makna': "Melambangkan kesederhanaan dan ketenangan.",
        'asal': "Solo dan Yogyakarta",
        'rekomendasi_acara': "Sesuai untuk acara formal, pertemuan keluarga, atau upacara adat."
    },
    'Batik Keraton': {
        'deskripsi': "Batik Keraton memiliki motif yang sakral dan hanya boleh dikenakan oleh keluarga kerajaan. Batik ini berasal dari daerah Yogyakarta dan Surakarta.",
        'makna': "Melambangkan kekuasaan dan kebangsawanan.",
        'asal': "Yogyakarta dan Surakarta",
        'rekomendasi_acara': "Sesuai untuk upacara kerajaan atau acara adat yang sakral."
    },
    'Batik Tambal': {
        'deskripsi': "Batik Tambal dibuat dengan menggabungkan potongan-potongan kain perca. Batik ini berasal dari daerah Yogyakarta.",
        'makna': "Melambangkan harapan untuk perbaikan dalam kehidupan.",
        'asal': "Yogyakarta",
        'rekomendasi_acara': "Cocok untuk acara budaya, festival, atau kegiatan seni."
    }
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
                batik_data = batik_info[predicted_class]
                html_info = f"""
                <div style='font-family:sans-serif'>
                    <h4>Informasi Batik {predicted_class}</h4>
                    <p><strong>Deskripsi:</strong> {batik_data['deskripsi']}</p>
                    <p><strong>Makna:</strong> {batik_data['makna']}</p>
                    <p><strong>Asal:</strong> {batik_data['asal']}</p>
                    <p><strong>Rekomendasi Acara:</strong> {batik_data['rekomendasi_acara']}</p>
                </div>
                """
                st.info(html_info, unsafe_allow_html=True)

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
                batik_data = batik_info[predicted_class]
                html_info = f"""
                <div style='font-family:sans-serif'>
                    <h4>Informasi Batik {predicted_class}</h4>
                    <p><strong>Deskripsi:</strong> {batik_data['deskripsi']}</p>
                    <p><strong>Makna:</strong> {batik_data['makna']}</p>
                    <p><strong>Asal:</strong> {batik_data['asal']}</p>
                    <p><strong>Rekomendasi Acara:</strong> {batik_data['rekomendasi_acara']}</p>
                </div>
                """
                st.info(html_info, unsafe_allow_html=True)
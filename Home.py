from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

logo = Image.open('assets/Logo.png')
st.set_page_config(
    page_title='KNOB 😱', 
    page_icon=logo, 
    layout='wide'
)

# Kustomisasi Page
st.header(':blue[Know Your Batik]', divider="gray")
st.write('''<div style="text-align: justify">
         <h3>Apa Itu KNOB (Know Your Batik)?</h3>
         Know Your Batik merupakan aplikasi yang memungkinkan Anda mendeteksi jenis batik berdasarkan gambar yang Anda unggah atau foto.
         </div>''', unsafe_allow_html=True)

st.write("<hr>", unsafe_allow_html=True)
st.write('''<div style="text-align: justify">
         <h3>Ayo Temukan Jenis Batikmu!</h3>
         </div>''', unsafe_allow_html=True)

# Button Find 
if st.button("Cari di sini"):
    st.switch_page("pages/1_Find.py")

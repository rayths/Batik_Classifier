import streamlit as st
from PIL import Image

logo = Image.open('assets/Logo.png')
st.set_page_config(
    page_title='KNOB - About Us', 
    page_icon=logo, 
    layout='wide'
)

st.header("About Us")

team_members = [
    {
        "name": "Deva Anjani Khayyuninafsyah", 
        "linkedin": "https://www.linkedin.com/in/deva-anjani-khayyuninafsyah-129684217/"
    },
    {
        "name": "Raid Muhammad Naufal",
        "linkedin": "https://www.linkedin.com/in/raidmnaufal/"
    },
    {
        "name": "Natasya Ega Lina Mabrun",
        "linkedin": "https://www.linkedin.com/in/natasya-ega-lina-marbun-a265492b2/"
    },
    {
        "name": "Feryadi Yulius",
        "linkedin": "https://www.linkedin.com/in/feryadi-yulius/"
    }
]

col1, col2 = st.columns(2)

for i, member in enumerate(team_members):
    col = col1 if i % 2 == 0 else col2
    col.markdown(f"""<h4>{member['name']}</h4>
                  <p><a href="{member['linkedin']}" target="_blank">LinkedIn Profile</a></p>
                  <hr>""", unsafe_allow_html=True)
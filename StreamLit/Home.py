import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import joblib
import PyPDF2
from PIL import Image
import base64
from io import BytesIO
import io

def app():
    st.markdown("""<style>#aboutApp {color: #31333f;font-size: 24px;direction: rtl;font-weight: bold;}</style>""", unsafe_allow_html=True)
    st.image('./deepfake2.jpg')
    st.write("### With this Application We Aim is to Detect and Validate Texts in Arabic or English, to Ensure it's not Artificially Generated by Large Language Models")
    st.markdown("<p id='aboutApp'>يساعدك هذا الموقع على إكتشاف النصوص التي تم صنعها بواسطة الذكاء الإصطناعي</p>", unsafe_allow_html=True)
    #   ********     FUNCTIONS

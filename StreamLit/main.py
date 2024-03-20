import streamlit as st
from streamlit_option_menu import option_menu
import altair as alt
import numpy as np
import pandas as pd
import joblib
import PyPDF2
from PIL import Image
import base64
from io import BytesIO
import io
from pyngrok import ngrok
import Home, EnglishDeepFakeDetection2, ArabicDeepFakeDetection
st.set_page_config(
        page_title="ArtificialTextDetection"
)



class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # ********   logo    ********
        def add_logo(logo_path, width, height):
            """Read and return a resized logo"""
            logo = Image.open(logo_path)
            modified_logo = logo.resize((width, height))
            return modified_logo

        def pil_to_base64(image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        ottawa_logo = add_logo(logo_path="ottawa.png", width=170, height=55)
        wakeb_logo = add_logo(logo_path="wakeb.png", width=170, height=40)

        # Centering images in Streamlit sidebar
        st.sidebar.markdown(
            "<div style='text-align: center; margin: 0; padding: 5;'>"
            "<img src='data:image/png;base64,{}' alt='Ottawa Logo' style='width: {}px; height: {}px; margin: 0; padding: 0;'>"
            "</div>".format(pil_to_base64(ottawa_logo), 170, 55),
            unsafe_allow_html=True
        )

        st.sidebar.markdown(
            "<div style='text-align: center; margin: 0; padding: 5;'>"
            "<img src='data:image/png;base64,{}' alt='Wakeb Logo' style='width: {}px; height: {}px; margin: 0; padding: 0;'>"
            "</div>".format(pil_to_base64(wakeb_logo), 170, 40),
            unsafe_allow_html=True
        )
        st.sidebar.markdown(
                        """
                        <style>
                            .sidebar .sidebar-content {
                                width: 670px;  
                            }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
        # ********   Menu    ********
        st.sidebar.markdown('<br>', unsafe_allow_html=True)
        with st.sidebar:        
            app = option_menu(
                menu_title='Artificial Text Detection',
                options=['Home','English Text','النص العربي','About'],
                icons=['house-fill','blockquote-right','blockquote-left','info-circle-fill'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                        "menu-title":{ "font-size": "20px"},
                        "icon": {"color": "black", "font-size": "18px"}, 
                        "nav-link": {"color":"black","font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#3be2ff"},
                        "nav-link-selected": {"background-color": "#027eab"}
                        }
                )


        if app == "Home":
            Home.app()
        if app == "English Text":
            EnglishDeepFakeDetection2.app()    
        if app == "النص العربي":
            ArabicDeepFakeDetection.app()        
             
    run() 

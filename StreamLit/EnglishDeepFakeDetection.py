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
from scipy.special import softmax
import langid
import pymongo
from datetime import datetime


def app():
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client["DfTxt"]
    collection = db["DfTxtDetectRecords"]

    #   ********     FUNCTIONS
    def detect_language(text):
        lang, _ = langid.classify(text)
        return lang
    def scan_pdf(file):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def extract_text_from_pdf(pdf_file):
        pdf_reader =  PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    
        return text

    def divide_into_500_word_items(text):
        if isinstance(text, list):
            text = ' '.join(text)

        word_list = text.split()
        items = []
        current_item = ''

        for word in word_list:
            if len(current_item.split()) < 500:
                current_item += ' ' + word
            else:
                items.append(current_item)
                current_item = word

        if current_item:
            items.append(current_item)

        return items

    def apply_prediction_on_dataX(data, model_selectbox, model_BERT):
        predictions_list, probabilities_list = [], []
        
        for te in data:
            te = [str(te)]
            if model_selectbox == "BERT":
                # prediction = model_BERT.predict(te)[0]
                prediction, raw_output = model_BERT.predict(te) 

                # Convert raw logits to probabilities using softmax
                probabilities = softmax(raw_output, axis=1)

                # Append probabilities as percentages with two decimal places
                probability_percentage = [f"{prob * 100:.2f}%" for prob in probabilities[0]]
                probabilities_list.append(probability_percentage)

                # Extract the probability corresponding to the predicted class
                if prediction[0] == 0:
                    pred = "Human"
                    pred_probability = probability_percentage[0]
                else:
                    pred = "Machine"
                    pred_probability = probability_percentage[1]
                    
                predictions_list.append(pred)

        dataframe = {'Text': data, 'Result': predictions_list, 'Probability': probabilities_list, 'Model': model_selectbox}
        df = pd.DataFrame(dataframe)
        df['Probability'] = [prob[prediction_idx] for prob, prediction_idx in zip(probabilities_list, df['Result'].apply(lambda x: 0 if x == 'Human' else 1))]
        st.write(df)
        
    def apply_prediction_on_data(data, model_selectbox, model_BERT, model_RoBERTa):
        predictions_list, probabilities_list = [], []
        total_human_probability, total_machine_probability = 0, 0

        for te in data:
            te = [str(te)]
            if model_selectbox == "BERT":
                prediction, raw_output = model_BERT.predict(te)
            else:
                prediction, raw_output = model_RoBERTa.predict(te)
                
            # Convert raw logits to probabilities using softmax
            probabilities = softmax(raw_output, axis=1)
            # Append probabilities as percentages with two decimal places
            probability_percentage = [prob * 100 for prob in probabilities[0]]
            probabilities_list.append(probability_percentage)
            
            # Extract the probability corresponding to the predicted class
            if prediction[0] == 0:
                pred = "Human"
                pred_probability = probability_percentage[0]
                total_human_probability += pred_probability
            else:
                pred = "Machine"
                pred_probability = probability_percentage[1]
                total_machine_probability += pred_probability

            predictions_list.append(pred)

        average_human_probability = total_human_probability / len(data) if len(data) > 0 else 0
        average_machine_probability = total_machine_probability / len(data) if len(data) > 0 else 0

        dataframe = {'Text': data, 'Result': predictions_list, 'Probability': probabilities_list, 'Model': model_selectbox}
        df = pd.DataFrame(dataframe)
        df['Probability'] = [prob[prediction_idx] for prob, prediction_idx in zip(probabilities_list, df['Result'].apply(lambda x: 0 if x == 'Human' else 1))]

        if average_human_probability > average_machine_probability:
            st.markdown(f"<h3 style='text-align: center; color: #469a50;'>{average_human_probability:.2f}% 'Human'</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center; color: #9a4646;'>{average_machine_probability:.2f}% 'Machine'</h3>", unsafe_allow_html=True)
        st.session_state.df = df
        st.write(df)
        
    #       ***     models   ***

    model_BERT = joblib.load('BERT-all.pkl')
    model_RoBERTa = joblib.load('Ro-BERTa-all.pkl')

    #      INTERFACE

    #    ********   SIDEBAR     *********
    st.sidebar.markdown('<br>', unsafe_allow_html=True)
    models_selectbox = st.sidebar.selectbox("Select Model", ["BERT", "RoBERTa"])

    #    ********   BODY     *********
    st.title("DeepFake Text Detection")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    text_input = st.text_area("or Enter the text", value="")

    # Submit button for text area or file upload
    btn_submit2check = st.button("Submit")
    if btn_submit2check:
        # Check if text or file is submitted
        if uploaded_file is not None or text_input != "":
            # Perform prediction and display results
            data_list = []
            if uploaded_file is not None:
                text = scan_pdf(uploaded_file)
                data_list = divide_into_500_word_items(text)
                st.text_area("Text", text, height=300)
            elif text_input != "":
                text = text_input
                data_list = divide_into_500_word_items(text)
            if len(data_list) > 0:
                apply_prediction_on_data(data_list, models_selectbox, model_BERT, model_RoBERTa)
                
            else:
                st.markdown("<p style='text-align: left; color: #88001b;padding:0;margin:0;'>Enter some text to work with . . .</p>", unsafe_allow_html=True)
                
        else:
            st.markdown("<p style='text-align: left; color: #88001b;padding:0;margin:0;'>Enter some text to work with . . .</p>", unsafe_allow_html=True)
            

    with st.expander("Provide Feedback"):        
        feedback_radio = st.radio("Was the prediction correct?", ["Yes", "No"], index=None)
        feedback_text = st.text_input("Do you have any other feedback? (optional)")
        feedback_button = st.button("Submit Feedback")

        if feedback_button:
            if not feedback_radio and not feedback_text:
                st.write("You didn't submit feedback.")
            else:
                # Convert DataFrame to dictionary
                df_dict = st.session_state.df.to_dict(orient='records')

                # Create a document to insert into MongoDB
                feedback_data = {
                    "chunks": [
                        {
                            "text": "",  # Placeholder value, modify this based on your needs
                            "result": ["", ""],
                            "Probability": "",
                            "model": ["", ""]
                        },
                    ],
                    "overall_feedback": {
                        "IsRight?": [feedback_radio],
                        "User_additional_feedback": feedback_text
                    },
                    "timestamp": datetime.utcnow()
                }

                # Modify the fields based on the DataFrame
                feedback_data["chunks"][0]["text"] = df_dict[0]["Text"]
                feedback_data["chunks"][0]["result"] = df_dict[0]["Result"]
                feedback_data["chunks"][0]["Probability"] = df_dict[0]["Probability"]
                feedback_data["chunks"][0]["model"] = df_dict[0]["Model"]

                # Insert the document into the MongoDB collection
                result = collection.insert_one(feedback_data)

                # Display a success message
                st.write("Feedback received, thank you for your feedback!")      